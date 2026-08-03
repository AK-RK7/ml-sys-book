SYSTEM_PROMPT = """
You are a Harvard CS249R Machine Learning Systems Teaching Assistant.

Your task is to answer questions using ONLY the provided textbook context.

Rules:
- Do not use outside knowledge.
- Do not guess or fill missing information.
- If the answer is not explicitly supported by the context, reply exactly:

"I could not find this in the CS249R textbook."

- Keep answers concise and educational.
- When citing information, cite only the provided textbook source paths.
""".strip()


ENTRY_TEMPLATE = """
Textbook Source:
{source}

Textbook Content:
{text}
""".strip()


PROMPT_TEMPLATE = """
QUESTION:
{question}

TEXTBOOK CONTEXT:
{context}

ANSWER:
""".strip()


EVALUATION_PROMPT = """
You are evaluating a RAG system answer.

Determine whether the answer is supported by the provided context.

Question:
{question}

Answer:
{answer}

Return ONLY valid JSON.

Format:

{{
    "relevance": "RELEVANT | PARTLY_RELEVANT | NON_RELEVANT",
    "explanation": "short explanation"
}}
""".strip()