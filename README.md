# PageIndex-OSS-Benchmark

A retrieval benchmark for **open-source PageIndex running locally** —
`PageIndexClient()` with no API key, flash indexing, no OCR.

We collect 62 questions over 34 PDFs, drawn from
[MMLongBench-Doc-V2](https://github.com/VectifyAI/MMLongBench-Doc-V2) and kept
to cases where the answer is a fact stated in running text: no charts, no tables, no figures, and no counting or arithmetic
on top of what was retrieved. The point is to isolate **retrieval and reading**
from everything else a document-QA system does — a wrong answer here is a
retrieval or extraction failure, not a reasoning one.

Scoped to what the OSS local path can actually ingest: every document was
verified to index under flash, and documents it refuses are excluded rather
than scored as failures. The set measures retrieval quality *within* PageIndex
OSS's reach, and says nothing about the documents outside it.

## Contents

| path | what |
|---|---|
| `documents/` | 34 source PDFs, 1,945 pages, 114 MB |
| `questions.json` | 62 questions with reference answers and provenance |
| `documents.json` | per-document page count, text density, gpt-5.6-luna indexing cost |
| `results.json` | measured accuracy and cost per model and effort setting |
| `run_variant.py` | index the PDFs, answer every question, price the run |
| `pi_bench.py` | indexing, metering and pricing helpers |
| `audit_lookup_gt.py` | re-run the reference-answer audit |

## Running it

```bash
pip install 'pageindex>=0.2.10.dev4'
export OPENAI_API_KEY=...

python run_variant.py gpt-5.6-luna high      # chat model, reasoning effort
```

The two arguments set the **chat** model only. Tree construction is separate:
flash parses the structure out of the PDF itself and calls a model only for
node summaries and the document description, using `index_model` — left at its
default, **gpt-5.6-luna**, for every tree shipped here. Override it if you want
a different indexer:

```python
PageIndexClient(index_model="gpt-5.6-luna", chat_model="gpt-5.6-luna")
```

First run indexes all 34 PDFs serially and caches the doc ids in
`indexed_docs.json`; later runs reuse them, so comparing chat models re-answers
over identical trees. Output is one JSON row per question with the response and
its measured token usage and cost.

## Scoring

Use [MMLongBench-Doc-V2](https://github.com/VectifyAI/MMLongBench-Doc-V2)'s
own judge — a semantic-equivalence call between the
reference answer and a response that never reads the source document, so it
cannot invent a new correct answer and its verdicts do not drift with whatever
system is being scored.

```bash
git clone https://github.com/VectifyAI/MMLongBench-Doc-V2
cd MMLongBench-Doc-V2
python -m eval.judge /path/to/predictions.json --out judged.json
```

`run_variant.py` writes rows in the shape the judge wants: `question`,
`answer`, `answer_format`, `response`.

## Results

Local PageIndex flash indexing with `index_model` at its default
**gpt-5.6-luna**, `responses()` protocol, scored by
[MMLongBench-Doc-V2](https://github.com/VectifyAI/MMLongBench-Doc-V2)'s judge.
The model column is the **chat** model; every row answers over the same
luna-built trees, so the differences are in reading, not indexing. Cost is the
answering call only, priced from litellm's cost map — indexing is one-off and
shared across all rows.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results-dark.png">
  <img src="results-light.png" alt="Accuracy against average cost per question. Each model forms a near-vertical reasoning-effort ladder; moving between models costs an order of magnitude a step.">
</picture>

<details>
<summary>The same numbers as a table</summary>

| model | effort | accuracy | avg $/question |
|---|---|---|---:|
| gpt-5.6-luna | none | 53/62 · 85.5% | $0.0031 |
| gpt-5.6-luna | low | 53/62 · 85.5% | $0.0033 |
| gpt-5.6-luna | medium | 57/62 · 91.9% | $0.0038 |
| gpt-5.6-luna | high | 60/62 · 96.8% | $0.0036 |
| gpt-5.6-terra | none | 56/62 · 90.3% | $0.0296 |
| gpt-5.6-terra | low | 59/62 · 95.2% | $0.0324 |
| gpt-5.6-terra | medium | 61/62 · 98.4% | $0.0303 |
| gpt-5.6-terra | high | 62/62 · 100.0% | $0.0325 |
| gpt-5.6-sol | none | 60/62 · 96.8% | $0.0759 |
| gpt-5.6-sol | low | 60/62 · 96.8% | $0.0817 |
| gpt-5.6-sol | medium | 62/62 · 100.0% | $0.0810 |
| gpt-5.6-sol | high | 62/62 · 100.0% | $0.0819 |

`results.json` carries them machine-readable.
</details>
