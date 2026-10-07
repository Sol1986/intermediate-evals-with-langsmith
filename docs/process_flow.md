Memorize this one sentence: **an evaluator can only grade what you traced, so name every step, and read the keys off your own trace.**

Everything else hangs off that. Here is the short version, in the order you'd use it.

**1. One function per step, each with `@traceable`.** If retrieval, tool calls, and generation are all mashed into one function, LangSmith records one blob and there is nothing to reach into. Separate functions become separate nodes in the tree.

**2. Give steps clear names.** The evaluator finds a step with `r.name == "retrieve"`, so the name is the address. Rename with `@traceable(name="retrieve")` if the function name isn't what you want.

**3. Look at the tree before you write the evaluator.** Open one trace in LangSmith and note how deep your step sits. Each level down is one `next(...)`. Your `target` wrapper adds a level, which is why we needed two.

  # Walk down the tree: target -> qa_pipeline -> retrieve
    qa_pipeline_run: Run = next(r for r in run.child_runs if r.name == "qa_pipeline")
    retrieve_run: Run = next(r for r in qa_pipeline_run.child_runs if r.name == "retrieve")

**4. Read the keys off the trace.** Bare list returned means `"output"`. Dictionary returned means your own keys. Never guess.

**5. Decide what you are measuring first.** Groundedness, correctness, and relevance are different metrics. Write the definition into the grader's instructions before you run anything.

**6. When a score surprises you, read the raw output before changing code.** The bug might be in the pipeline, the grader, or your expectation.

A memory hook: **Name it, Find it, Key it, Define it.** Name the steps, find them in the tree, read the keys, define the metric.

Or the office building version: the trace is the building directory. Steps need room numbers (names), you walk floor by floor (`next`), you read the label on the folder (keys), and you know what you are inspecting for (the metric).

If you only keep one habit, keep step 3. Opening the trace and looking at the tree first prevents most of the errors we hit.