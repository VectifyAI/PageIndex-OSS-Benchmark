"""Answer a question set with a chosen chat model + reasoning effort, priced.

    python run_variant.py                          # gpt-5.6-luna, high
    python run_variant.py gpt-5.6-sol
    python run_variant.py gpt-5.6-luna low --set data/lookup10.json

Indexing is flash-only and cached in data/indexed_docs.json, shared across
variants — the index is a property of the document, not of the chat model, so
every variant answers over exactly the same trees and index cost is paid once.

Writes data/<set>.<model>-<effort>.json, ready for the benchmark's own judge:

    cd MMLongBench-Doc-V2
    python -m eval.judge ../data/lookup10.gpt-5.6-luna-high.json \
        --out ../data/lookup10.luna-high.judged.json
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import statistics

import pi_bench
from pageindex import PageIndexClient


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", nargs="?", default="gpt-5.6-luna")
    ap.add_argument("effort", nargs="?", default="high",
                    choices=["none", "low", "medium", "high", "default"])
    ap.add_argument("--set", dest="qset",
                    default="questions.json" if os.path.exists("questions.json")
                    else "data/lookup10.json")
    ap.add_argument("--out", default=None,
                    help="output path; default data/<set-stem>.<model>-<effort>.json")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--max_turns", type=int, default=None,
                    help="cap agent turns — the main cost lever, since every "
                         "turn resends the whole transcript")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.qset))[0]
    outdir = "data" if os.path.isdir("data") else "."
    out_path = args.out or f"{outdir}/{stem}.{args.model}-{args.effort}.json"
    reasoning = None if args.effort == "default" else {"effort": args.effort}

    rows = json.load(open(args.qset))
    client = PageIndexClient(chat_model=args.model)

    # Serial on purpose: flash opens an event loop per call and utils' client
    # global does not survive that. See pi_bench.Meter.
    cache = pi_bench.load_cache()
    for row in rows:
        pi_bench.index_doc(client, row["doc_id"], cache)
    todo = [r for r in rows if r["doc_id"] in cache]
    skipped = len(rows) - len(todo)

    def ask(row):
        env = client.responses(row["question"], doc_id=cache[row["doc_id"]]["doc_id"],
                               reasoning=reasoning, max_turns=args.max_turns)
        u = env["usage"]
        d = u["input_tokens_details"]
        return {**row,
                "response": pi_bench.answer_text(env),
                "chat_model": args.model, "effort": args.effort,
                "tool_calls": pi_bench.tool_calls(env),
                "input": u["input_tokens"],
                "cached": d["cached_tokens"],
                "cache_write": d["cache_write_tokens"],
                "output": u["output_tokens"],
                "reasoning_tokens": u["output_tokens_details"]["reasoning_tokens"],
                "index_cost": cache[row["doc_id"]]["index_cost"],
                "ask_cost": round(pi_bench.price(args.model, u), 6)}

    # Answering parallelises safely: local_chat builds its own AsyncOpenAI per
    # call and closes it, so nothing is shared across loops.
    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        preds = list(ex.map(ask, todo))

    json.dump(preds, open(out_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n{'doc':<32} {'calls':>5} {'input':>9} {'cached':>8} {'out':>6} "
          f"{'ask $':>8}")
    for p in sorted(preds, key=lambda x: -x["ask_cost"]):
        print(f"{p['doc_id'][:30]:<32} {p['tool_calls']:>5} {p['input']:>9,} "
              f"{p['cached']:>8,} {p['output']:>6,} {p['ask_cost']:>8.4f}")

    ask_costs = [p["ask_cost"] for p in preds]
    priced = [p["index_cost"] for p in preds if p["index_cost"] is not None]
    total = sum(ask_costs)
    print(f"\n{args.model} effort={args.effort} | {len(preds)} answered"
          + (f", {skipped} skipped (flash found no structure)" if skipped else ""))
    print(f"ask   ${total:.4f} total | mean ${statistics.mean(ask_costs):.4f} "
          f"| median ${statistics.median(ask_costs):.4f} | max ${max(ask_costs):.4f}")
    print(f"index ${sum(priced):.4f} total (one-off, shared by every variant)"
          + (f" — {len(preds) - len(priced)} indexed before metering, unpriced"
             if len(priced) < len(preds) else ""))
    # A cached doc keeps whatever mode built it. Say so: trees from a
    # different pipeline are not comparable to the flash ones.
    modes = collections.Counter(cache[p["doc_id"]]["mode"] for p in preds)
    if set(modes) - {"flash"}:
        print("index modes: " + ", ".join(f"{m}×{n}" for m, n in modes.items())
              + "  (non-flash trees came from the cache, not this run)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
