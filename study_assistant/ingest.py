import json
import pickle
import re
from pathlib import Path

from study_assistant.minsearch import Index

BASE_DIR = Path(__file__).resolve().parent.parent

BOOK_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "cs249r_book"
    / "book"
    / "quarto"
    / "contents"
)

PROCESSED_DIR = BASE_DIR / "data" / "processed"

DOCUMENTS_PATH = PROCESSED_DIR / "documents.json"
INDEX_PATH = PROCESSED_DIR / "index.pkl"

CHUNK_SIZE = 1000


def get_source_files():
    """Return all Quarto and Markdown source files."""

    files = list(BOOK_DIR.rglob("*.qmd"))
    files.extend(BOOK_DIR.rglob("*.md"))

    return sorted(files)


def clean_text(text: str) -> str:

    # Remove YAML header
    text = re.sub(
        r"\A\s*---.*?---",
        "",
        text,
        flags=re.DOTALL,
    )


    # Remove Quarto directives
    text = re.sub(
        r":::\s*\{.*?\}",
        "",
        text,
    )

    text = re.sub(
        r":::",
        "",
        text,
    )


    # Remove HTML blocks
    text = re.sub(
        r"<[^>]+>",
        "",
        text,
    )


    # Remove code fences
    text = re.sub(
        r"```.*?```",
        "",
        text,
        flags=re.DOTALL,
    )


    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )


    return text.strip()

def chunk_document(text: str, source: str):
    """Split a cleaned document into retrieval chunks."""

    chunks = []

    paragraphs = text.split("\n\n")

    current = ""

    # stable document id based on relative path
    doc_id = (
        source.replace("/", "_")
              .replace(".qmd", "")
              .replace(".md", "")
    )

    chunk_no = 0

    for para in paragraphs:

        if len(current) + len(para) <= CHUNK_SIZE:
            current += "\n" + para

        else:
            if current.strip():
                chunks.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_{chunk_no:04d}",
                        "text": current.strip(),
                        "source": source,
                        "volume": source.split("/")[0],
                    }
                )
                chunk_no += 1

            current = para

    if current.strip():
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{chunk_no:04d}",
                "text": current.strip(),
                "source": source,
                "volume": source.split("/")[0],
            }
        )

    return chunks

def build_documents():
    """Create retrieval documents."""

    documents = []

    for file in get_source_files():

        text = file.read_text(
            encoding="utf-8"
        )

        text = clean_text(text)

        documents.extend(
            chunk_document(
                text=text,
                source=file.relative_to(BOOK_DIR).as_posix(),
            )
        )

    return documents


def build_index():
    """Build and persist the retrieval index."""

    documents = build_documents()

    index = Index(
        text_fields=["text"],
        keyword_fields=[
            "doc_id",
            "chunk_id",
            "source",
            "volume",
        ],
    )

    index.fit(documents)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DOCUMENTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            documents,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with INDEX_PATH.open("wb") as f:

        pickle.dump(index, f)

    print(f"Indexed {len(documents)} chunks.")
    print(f"Saved documents -> {DOCUMENTS_PATH}")
    print(f"Saved index -> {INDEX_PATH}")

    return index


def load_index():
    try:
        with INDEX_PATH.open("rb") as f:
            index = pickle.load(f)

        if not hasattr(index, "vectorizers"):
            raise ValueError("Outdated index")

        return index

    except Exception:
        return build_index()

if __name__ == "__main__":
    build_index()