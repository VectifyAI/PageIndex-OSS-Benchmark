"""Shared plumbing for the MMLongBench-Doc-V2 probe runs on local PageIndex.

Three things the SDK does not hand you, collected here so the runners stay thin:

Indexing usage. submit_document() returns {doc_id, name} and nothing else —
utils.llm_completion / llm_acompletion drop response.usage on the floor. Meter
counts it by owning the client objects utils would otherwise create lazily.

Cross-loop clients. utils keeps _openai_async_client as a module global and
reuses it forever, but flash indexing opens a fresh event loop per call
(flash/api.py asyncio.run, twice). An httpx pool binds to the loop that first
used it, so a reused client hits dead connections — "Connection error" retry
storms, and a hard deadlock if several run in threads. Meter.install() reseats
both globals, so call it before every submit_document and index serially.

Pricing. Rates come from litellm's cost map, never hardcoded. `cached` and
`cache_write` are subsets carved out of input_tokens, not additions, so the
three input tiers are priced separately and fresh input is the remainder.

Requires pageindex >= 0.2.10.dev4 (native `reasoning=` on responses(),
index_model / chat_model config keys).
"""
import collections
import json
import os
import threading

import litellm

# Works from either root: inside the released dataset (documents/ alongside
# questions.json) or from the parent project against the full corpus.
# $PI_DOCS overrides both.
DOCS = os.environ.get("PI_DOCS") or (
    "documents" if os.path.isdir("documents") else "MMLongBench-Doc/data/documents")
INDEX_CACHE = os.environ.get("PI_INDEX_CACHE") or (
    "indexed_docs.json" if os.path.isdir("documents") else "data/indexed_docs.json")


# ── pricing ────────────────────────────────────────────────────────────

def price(model, usage):
    """Cost in USD of one Responses-dialect usage dict."""
    p = litellm.model_cost.get(model)
    if p is None:
        raise KeyError(f"{model} is not in litellm's cost map — cannot price it")
    d = usage.get("input_tokens_details", {})
    cached = d.get("cached_tokens") or 0
    write = d.get("cache_write_tokens") or 0
    fresh = max(0, usage["input_tokens"] - cached - write)
    return (fresh * (p.get("input_cost_per_token") or 0)
            + cached * (p.get("cache_read_input_token_cost") or 0)
            + write * (p.get("cache_creation_input_token_cost") or 0)
            + usage["output_tokens"] * (p.get("output_cost_per_token") or 0))


def price_chat_dialect(model, prompt, completion, cached=0, write=0):
    """Same maths for the chat.completions field names used while indexing."""
    return price(model, {"input_tokens": prompt, "output_tokens": completion,
                         "input_tokens_details": {"cached_tokens": cached,
                                                  "cache_write_tokens": write}})


# ── indexing usage meter ───────────────────────────────────────────────

class Meter:
    """Counts indexing tokens per model, and reseats utils' client globals.

    Install before each submit_document: the fresh async client is what keeps
    flash's per-call event loops from reusing a pool bound to a closed one.
    """

    def __init__(self):
        self.usage = collections.defaultdict(collections.Counter)
        self._lock = threading.Lock()

    def _record(self, model, response):
        u = getattr(response, "usage", None)
        if u is None:
            return
        d = getattr(u, "prompt_tokens_details", None)
        with self._lock:
            c = self.usage[model]
            c["in"] += u.prompt_tokens or 0
            c["out"] += u.completion_tokens or 0
            c["cached"] += getattr(d, "cached_tokens", 0) or 0
            c["write"] += getattr(d, "cache_write_tokens", 0) or 0
            c["calls"] += 1

    def _wrap(self, client, is_async):
        create = client.chat.completions.create
        if is_async:
            async def wrapped(*a, **kw):
                r = await create(*a, **kw)
                self._record(kw.get("model", "?"), r)
                return r
        else:
            def wrapped(*a, **kw):
                r = create(*a, **kw)
                self._record(kw.get("model", "?"), r)
                return r
        client.chat.completions.create = wrapped
        return client

    def install(self):
        import openai
        from pageindex import utils
        utils._openai_sync_client = self._wrap(openai.OpenAI(max_retries=0), False)
        utils._openai_async_client = self._wrap(openai.AsyncOpenAI(max_retries=0), True)

    def cost(self):
        return sum(price_chat_dialect(m, c["in"], c["out"], c["cached"], c["write"])
                   for m, c in self.usage.items())

    def tokens(self):
        return sum(c["in"] + c["out"] for c in self.usage.values())

    def reset(self):
        with self._lock:
            self.usage.clear()


# ── document indexing ──────────────────────────────────────────────────

def load_cache(path=INDEX_CACHE):
    """Load the doc-id cache, upgrading legacy string entries.

    Early runs stored a bare doc_id per PDF. Those carry no cost record, so
    index_cost stays None rather than being invented as 0 — totals report how
    many are unpriced instead of quietly understating.
    """
    raw = json.load(open(path)) if os.path.exists(path) else {}
    return {name: (entry if isinstance(entry, dict)
                   else {"doc_id": entry, "mode": "unknown",
                         "index_tokens": None, "index_cost": None})
            for name, entry in raw.items()}


# Reasons a specific PDF cannot be indexed, as opposed to the run being
# misconfigured (bad key, unknown mode, missing file). These mean "skip this
# document"; anything else must surface.
UNINDEXABLE = (
    "could not extract a structure",   # flash: no detectable hierarchy
    "PDF has no content",              # no text layer at all (scanned)
    "could not read PDF",              # corrupt or unparseable
    "produced no structure",           # standard's equivalent of the first
)


def index_doc(client, name, cache, path=INDEX_CACHE, allow_standard=False):
    """Index one PDF, flash-only by default. Returns a cache entry or None.

    Flash reads structure off the PDF and gives up on documents with no
    detectable hierarchy; those are skipped rather than silently falling back
    to standard, which is a different (and far more expensive) pipeline whose
    numbers do not belong in the same table.
    """
    from pageindex.errors import PageIndexAPIError

    if name in cache:
        # Failures are cached too, as an entry with no doc_id, so a rerun does
        # not pay to rediscover that a PDF is unindexable.
        return cache[name] if cache[name].get("doc_id") else None
    meter = Meter()
    modes = ("flash", "standard") if allow_standard else ("flash",)
    for mode in modes:
        meter.reset()
        meter.install()
        try:
            doc = client.submit_document(os.path.join(DOCS, name), wait=True,
                                         mode=mode)
        except PageIndexAPIError as e:
            reason = next((r for r in UNINDEXABLE if r in str(e)), None)
            if reason is None:
                raise
            if mode == modes[-1]:
                cache[name] = {"doc_id": None, "mode": None,
                               "index_tokens": None, "index_cost": None,
                               "skipped": reason}
                json.dump(cache, open(path, "w"), indent=2)
                print(f"SKIP {name}: {reason}", flush=True)
                return None
            continue
        cache[name] = {"doc_id": doc["doc_id"], "mode": mode,
                       "index_tokens": meter.tokens(),
                       "index_cost": round(meter.cost(), 6)}
        json.dump(cache, open(path, "w"), indent=2)
        print(f"indexed {name} ({mode}) {meter.tokens():,} tok "
              f"${cache[name]['index_cost']:.4f}", flush=True)
        return cache[name]


# ── answering ──────────────────────────────────────────────────────────

def answer_text(envelope):
    """The assistant text out of a Responses envelope."""
    parts = [c["text"]
             for item in envelope["output"] if item.get("type") == "message"
             for c in item.get("content", []) if c.get("text")]
    return "\n".join(parts).strip()


def tool_calls(envelope):
    return sum(1 for i in envelope["items"] if i.get("type") == "function_call")
