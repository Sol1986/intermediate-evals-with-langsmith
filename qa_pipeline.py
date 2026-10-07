from dotenv import load_dotenv
load_dotenv()

import wikipedia as wp
from openai import OpenAI
from langsmith import traceable, wrappers

oai_client = wrappers.wrap_openai(OpenAI())

wp.set_user_agent("eval-intermediate-steps/0.1 (sfproject@gmail.com)")



@traceable
def retrieve(question: str) -> list[dict]:
    """Search Wikipedia and return up to 3 passages as dictionaries."""
    docs = []
    for title in wp.search(question, results=3):
        try:
            text = wp.summary(title, sentences=5)
            docs.append({"page_content": text, "metadata": {"title": title}})
        except Exception:
            continue  # skip disambiguation pages and missing pages
    return docs


@traceable
def generate(question: str, docs: list[dict]) -> str:
    """Answer a question using the retrieved passages as context.""" 
    context = "\n\n".join([doc["page_content"] for doc in docs])
    response = oai_client.chat.completions.create(
        model="gpt-5.4-mini",  # swap for any model you have access to
        messages=[
    {
        "role": "system",
        "content": (
            "Answer using only the provided context. If the context does not "
            "contain the answer, reply exactly: I don't know based on the "
            "provided context."
        ),
    },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


@traceable
def qa_pipeline(question: str) -> dict:
    """Run retrieve then generate, and return the answer in a dictionary."""
    docs = retrieve(question)
    answer = generate(question, docs)
    return {"answer": answer}


if __name__ == "__main__":
    questions = [
        "Who invented the telephone?",
        "What is the capital of Australia?",
        "Who won the 2031 FIFA World Cup?",  # the trick question
    ]
    for q in questions:
        print(q)
        print(qa_pipeline(q)["answer"])
        print()