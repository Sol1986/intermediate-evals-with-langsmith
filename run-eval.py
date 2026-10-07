from dotenv import load_dotenv

load_dotenv()

from langsmith import Client, evaluate
from langsmith.schemas import Run

# Import the pipeline and evaluator from the files you already wrote.
# Hyphenated filenames can't be imported with a normal import statement,
# so we load them with importlib.
import importlib

pipeline_module = importlib.import_module("qa_pipeline")
evaluator_module = importlib.import_module("qa_evaluator")

DATASET_NAME: str = "wiki-qa-learning-set-3"

QUESTIONS: list[str] = [
    "Who invented the telephone?",
    "What is the capital of Australia?",
    "Who won the 2031 FIFA World Cup?",  # the trick question
]


def build_dataset(client: Client) -> None:
    """Create the dataset once. Skips creation if it already exists."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        return
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        inputs=[{"question": q} for q in QUESTIONS],
        dataset_id=dataset.id,
    )


def target(inputs: dict) -> dict:
    """Wrapper LangSmith calls once per example. Returns the pipeline's answer."""
    return pipeline_module.qa_pipeline(inputs["question"])


def no_hallucination(run: Run) -> bool:
    """Score one run by delegating to the evaluator we wrote."""
    return evaluator_module.no_hallucination(run)


if __name__ == "__main__":
    client: Client = Client()
    build_dataset(client)
    evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[no_hallucination],
        experiment_prefix="wiki-qa-grounding",
    )