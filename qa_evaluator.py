from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from langsmith.schemas import Run
from pydantic import BaseModel, Field

# Plain client on purpose. The grader's own calls should not be traced
# as part of the pipeline we are grading.
grader_client: OpenAI = OpenAI()


class GradeHallucinations(BaseModel):
    """The grader's verdict on one answer."""

    is_grounded: bool = Field(
        ...,
        description="True if the answer is supported by the set of facts, False otherwise.",
    )


def no_hallucination(run: Run) -> bool:
    """Check whether the answer is grounded in the retrieved passages.

    Returns True if there is no hallucination, False otherwise.
    """
    # Walk down the tree: target -> qa_pipeline -> retrieve
    qa_pipeline_run: Run = next(r for r in run.child_runs if r.name == "qa_pipeline")
    retrieve_run: Run = next(r for r in qa_pipeline_run.child_runs if r.name == "retrieve")

    # retrieve returned a bare list, so LangSmith stored it under "output"
    retrieved_content: str = "\n\n".join(
        doc["page_content"] for doc in retrieve_run.outputs["output"]
    )
    answer: str = run.outputs["answer"]

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a grader. If the generation only says it does not know or "
                "cannot find the answer based on the provided context, and states no "
                "other facts, set is_grounded to True. Otherwise, set is_grounded to "
                "True only if every claim in the generation is supported by the set "
                "of facts."
            ),
        },
        {
            "role": "user",
            "content": f"Set of facts:\n{retrieved_content}\n\nLLM generation: {answer}",
        },
    ]

    completion = grader_client.beta.chat.completions.parse(
        model="gpt-5.4-mini",  # same model name as before, swap if needed
        messages=messages,
        response_format=GradeHallucinations,
        temperature=0,
    )
    grade: GradeHallucinations = completion.choices[0].message.parsed
    return grade.is_grounded