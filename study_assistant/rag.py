import hashlib
import tiktoken
from time import time

from rank_bm25 import BM25Okapi

from study_assistant.config import client
from study_assistant.ingest import load_index

from study_assistant.prompts import (
    SYSTEM_PROMPT,
    ENTRY_TEMPLATE,
    PROMPT_TEMPLATE,
)

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

MODEL = "llama-3.1-8b-instant"

ANSWER_MODEL = MODEL
EVALUATION_MODEL = MODEL

index = load_index()

documents_cache = index.documents

for doc in documents_cache:
    if "content" not in doc and "text" in doc:
        doc["content"] = doc["text"]

bm25 = BM25Okapi(
    [
        (doc.get("content") or doc.get("text", "")).lower().split()
        for doc in documents_cache
    ]
)

BOOST = {"text": 0.25}

VECTOR_WEIGHT = 0.85
BM25_WEIGHT = 0.15


def retrieval_penalty(source):

    blocked = [
        "frontmatter",
        "acknowledgements",
        "appendix",
        "glossary",
    ]

    return 0.5 if any(x in source.lower() for x in blocked) else 1.0


def rewrite_query(query, model=MODEL):

    prompt = f"""
    Rewrite this question into a short search query.

    Rules:
    - Return only the rewritten query.
    - No explanation.
    - No introduction.
    - Maximum 15 words.

    Question:
    {query}
    """

    rewritten, _ = llm(prompt, model)

    return rewritten.strip()


def hybrid_search(query, num_results=5):

    vector_results = index.search(
        query=query,
        filter_dict={},
        boost_dict=BOOST,
        num_results=num_results,
    )

    bm25_scores = bm25.get_scores(query.lower().split())

    bm25_ids = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:num_results]

    candidates = {}

    # Vector retrieval
    for rank, doc in enumerate(vector_results):

        source = doc.get("source", "unknown")

        candidates[source] = {
            **doc,
            "source": source,
            "score": VECTOR_WEIGHT
            * (1 / (rank + 1))
            * retrieval_penalty(source),
        }

    max_score = max(bm25_scores) if len(bm25_scores) else 0

    # BM25 retrieval
    for idx in bm25_ids:

        doc = documents_cache[idx]

        score = (
            BM25_WEIGHT * (bm25_scores[idx] / max_score)
            if max_score > 0
            else 0
        )

        if doc["source"] in candidates:

            candidates[doc["source"]]["score"] += score

        else:

            candidates[doc["source"]] = {**doc, "score": score}

    return sorted(candidates.values(), key=lambda x: x["score"], reverse=True)[
        :num_results
    ]


def build_prompt(question, documents):

    context = ""

    for doc in documents:

        text = doc.get("content") or doc.get("text", "")

        context += ENTRY_TEMPLATE.format(
            source=doc["source"], text=text[:3000]
        )

        context += "\n\n"

    return PROMPT_TEMPLATE.format(
        question=question,
        context=context.strip(),
    )


def llm(prompt, model):

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    usage = response.usage

    return (
        response.choices[0].message.content,
        {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    )


DEFAULT_PRICING = {
    "input": 0.00000005,
    "output": 0.00000008,
}

MODEL_PRICING = {
    "llama-3.1-8b-instant": DEFAULT_PRICING,
}


def calculate_cost(tokens, model):

    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)

    prompt_cost = tokens.get("prompt_tokens", 0) * pricing["input"]
    completion_cost = tokens.get("completion_tokens", 0) * pricing["output"]

    return prompt_cost + completion_cost


def rerank_documents(documents, query):

    pairs = [
        [doc.get("content") or doc.get("text", ""), query]
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [doc for doc, score in ranked[:5]], True


def get_prompt_version():

    content = SYSTEM_PROMPT + ENTRY_TEMPLATE + PROMPT_TEMPLATE

    return hashlib.md5(content.encode()).hexdigest()[:8]


RETRIEVAL_METHOD = "hybrid"

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):

    return len(encoding.encode(text))


def rag(question, model):

    start = time()

    rewritten = rewrite_query(question)

    documents = hybrid_search(rewritten)

    documents, reranker_used = rerank_documents(documents, rewritten)

    if not documents:

        return {
            "answer": "I could not find this in the CS249R textbook.",
            "sources": [],
            "rewritten_query": rewritten,
            "retrieved_chunks": 0,
            "context_tokens": 0,
            "response_time": time() - start,
            "model_used": model,
            "retrieval_method": RETRIEVAL_METHOD,
            "reranker_used": reranker_used,
            "prompt_version": get_prompt_version(),
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "relevance": "UNKNOWN",
            "cost": 0,
        }

    retrieval_scores = [float(d.get("score", 0)) for d in documents]

    prompt = build_prompt(question, documents)

    answer, tokens = llm(prompt, model=model)

    context_tokens = sum(
        count_tokens(doc.get("content") or doc.get("text", ""))
        for doc in documents
    )

    return {
        "answer": answer,
        "sources": sorted({d.get("source", "unknown") for d in documents}),
        "rewritten_query": rewritten,
        "top_retrieval_score": max(retrieval_scores),
        "avg_retrieval_score": (sum(retrieval_scores) / len(retrieval_scores)),
        "retrieval_method": RETRIEVAL_METHOD,
        "retrieved_chunks": len(documents),
        "reranker_used": reranker_used,
        "prompt_version": get_prompt_version(),
        "context_tokens": context_tokens,
        "relevance": None,
        "response_time": time() - start,
        "model_used": model,
        "total_tokens": tokens["total_tokens"],
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "cost": calculate_cost(tokens, model),
    }
