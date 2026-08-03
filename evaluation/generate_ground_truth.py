import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm
from pydantic import BaseModel

import random

import time
from openai import RateLimitError

class Question(BaseModel):
    question: str


class QuestionList(BaseModel):
    questions: list[Question]

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "documents.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "ground_truth_retrieval.csv"
)

MODEL = "openai/gpt-oss-120b"


PROMPT_TEMPLATE = """
You are generating evaluation questions for a RAG system.

Read the passage below and generate exactly TWO questions that can be answered using only this passage.

Return ONLY valid JSON in this exact format:

[
  {{
    "question": "..."
  }},
  {{
    "question": "..."
  }}
]

Do not include markdown, explanations, or any other text.

PASSAGE

{passage}
""".strip()


def load_documents():
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


MAX_RETRIES = 5
RETRY_DELAY = 200

def generate_questions(document):
    prompt = PROMPT_TEMPLATE.format(
        passage=document["text"]
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": prompt}],
            )
            break

        except RateLimitError:
            if attempt == MAX_RETRIES:
                print(
                    f"Skipping chunk {document['chunk_id']} after "
                    f"{MAX_RETRIES} rate-limit retries."
                )
                return []

            print(
                f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                f"Waiting {RETRY_DELAY} seconds..."
            )
            time.sleep(RETRY_DELAY)

    text = response.output_text.strip()

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1:
        print("Invalid response:")
        print(text)
        return []

    try:
        questions = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        print("Failed to parse JSON:")
        print(text)
        return []

    records = []

    for q in questions:
        records.append(
            {
                "doc_id": document["doc_id"],
                "chunk_id": document["chunk_id"],
                "source": document["source"],
                "question": q["question"],
            }
        )

    return records


def main():
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        processed = set(existing["chunk_id"])
    else:
        processed = set()
    documents = load_documents()

    random.seed(42)
    documents = random.sample(
        documents,
        min(500, len(documents))
    )

    for document in tqdm(documents):

        if document["chunk_id"] in processed:
            continue

        questions = generate_questions(document)

        if not questions:
            continue

        df = pd.DataFrame(questions)

        df.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=not OUTPUT_PATH.exists(),
            index=False,
        )

        processed.add(document["chunk_id"])

    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()