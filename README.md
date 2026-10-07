# Evaluating Intermediate Steps in LangSmith

A small Wikipedia question answerer with an evaluator that checks one of its **intermediate steps**. The evaluator doesn't only look at the final answer. It reaches into the trace, pulls out the passages the retriever returned, and asks a grader model whether the answer is grounded in them.

I built this to learn how LangSmith evaluators can read the steps inside a run, not just the inputs and outputs at the edges.

## Summary: I thought my evaluator caught a hallucination. It caught something else.

I gave the pipeline three questions. One was a trick: **"Who won the 2031 FIFA World Cup?"** The tournament hasn't happened, so I expected the model to either invent a winner, which the evaluator would catch, or admit it didn't know.

The evaluator returned **False**, and at first I took that as a caught hallucination. Then I read the trace. The model had said:

> The 2031 FIFA Women's World Cup has not been held yet, so there is no winner as of now.

That answer is correct. The evaluator flagged it anyway, and it was right to, because of what I had asked it to check.

**Groundedness and correctness are different metrics.**

- **Groundedness:** is the answer supported by the retrieved source material?
- **Correctness:** is the answer true in the real world?

My grader asked "is every claim supported by these passages?" The passages said nothing about the 2031 tournament, so the model's true statement had no support and failed. The grader worked as written. I had called the metric "hallucination" when what it measured was groundedness.

For a RAG system that must answer only from your own documents, such as a policy bot or support assistant, groundedness is the right metric, and a true answer with no source should fail. If you care about real-world truth, you need a different evaluator that compares against a reference answer. Choose the metric before you write the grader.

**The fix needed changes on both sides.** Editing only the grader to accept "I don't know" didn't work, because the model's answer still made two unsupported factual claims. So I:

1. Tightened the pipeline's system prompt so the model answers only from the context, and otherwise replies exactly: *"I don't know based on the provided context."*
2. Added an explicit first rule to the grader: an answer that only says it doesn't know, and states no other facts, counts as grounded.

After rerunning, the 2031 question flipped to **True**.

**The lesson:** an evaluator is only as good as its definition. When a score surprises you, read the actual output before you change anything. The bug can be in the pipeline, the grader, or your expectation.

## How it works

```
target (run-eval.py)
└── qa_pipeline
    ├── retrieve   → searches Wikipedia, returns up to 3 passages
    └── generate   → answers using only those passages
```

Every step is decorated with `@traceable`, so each one appears as its own node in the LangSmith trace. The evaluator receives the root `run` and walks down the tree by name, one `next(...)` per level:

```python
qa_pipeline_run = next(r for r in run.child_runs if r.name == "qa_pipeline")
retrieve_run = next(r for r in qa_pipeline_run.child_runs if r.name == "retrieve")
```

It joins the retrieved passages (stored under `"output"`, because `retrieve` returns a bare list), reads the final answer from `run.outputs["answer"]`, and sends both to a grader model that returns a structured `is_grounded` boolean.

## Files

| File | What it does |
|---|---|
| [qa_pipeline.py](qa_pipeline.py) | The traced pipeline: `retrieve`, `generate`, `qa_pipeline`. Run it directly to print answers to the three questions. |
| [qa_evaluator.py](qa_evaluator.py) | The `no_hallucination` evaluator. Finds the `retrieve` step in the trace and grades the answer for groundedness. |
| [run-eval.py](run-eval.py) | Creates the LangSmith dataset (once) and runs the experiment with the evaluator. |
| [docs/process_flow.md](docs/process_flow.md) | Short checklist for evaluating intermediate steps: name it, find it, key it, define it. |
| [docs/explanation.md](docs/explanation.md) | Step-by-step notes on building the pipeline, evaluator, and evaluation run. |

## Setup

Requires Python 3.10+, an OpenAI API key, and a LangSmith account.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-key
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=your-project-name
```

Both the pipeline and the grader use `gpt-5.4-mini`. Change the model name in `qa_pipeline.py` and `qa_evaluator.py` to any model you have access to.

## Running it

Try the pipeline on its own:

```bash
python qa_pipeline.py
```

Run the evaluation:

```bash
python run-eval.py
```

This creates the `wiki-qa-learning-set-3` dataset if it doesn't exist and runs an experiment prefixed `wiki-qa-grounding`. Open the experiment in LangSmith to see the `no_hallucination` score next to each question. The scores appear as feedback on each trace, not as separate traces.

## Takeaways

- **An evaluator can only grade what you traced.** Put each step in its own `@traceable` function.
- **The step name is its address.** The evaluator finds steps by `r.name`.
- **Look at the trace tree before writing the evaluator.** Each level down is one `next(...)`. The `target` wrapper adds a level.
- **Read output keys off the trace, don't guess.** A bare list is stored under `"output"`; a dict keeps your keys.
- **Define the metric first.** Groundedness, correctness, and relevance are different questions.
- **When a score surprises you, read the raw output** before touching the code.
