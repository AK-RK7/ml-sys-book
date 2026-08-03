from uuid import UUID, uuid4

from flask import Flask, jsonify, request
from psycopg2 import Error

from study_assistant.rag import rag
from study_assistant.db import (
    get_connection,
    init_db,
    save_conversation,
    save_feedback,
)

app = Flask(__name__)

# Initialize database tables
init_db()


@app.post("/question")
def question():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    question_text = data.get("question")

    if not question_text:
        return jsonify({"error": "Question is required"}), 400

    # Read model from request or use default
    model = data.get("model", "llama-3.1-8b-instant")

    conversation_id = str(uuid4())

    try:
        result = rag(question_text, model)

    except Exception as e:
        return (
            jsonify(
                {
                    "error": "RAG pipeline failed",
                    "details": str(e),
                }
            ),
            500,
        )

    save_conversation(
        conversation_id=conversation_id,
        question=question_text,
        answer_data=result,
    )

    return jsonify(
        {
            "conversation_id": conversation_id,
            "question": question_text,
            "rewritten_query": result.get("rewritten_query"),
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            # LLM metrics
            "model": result.get("model_used"),
            "response_time": result.get("response_time"),
            "prompt_version": result.get("prompt_version"),
            # Retrieval metrics
            "retrieval_method": result.get("retrieval_method"),
            "retrieved_documents": result.get("retrieved_documents"),
            "retrieved_chunks": result.get("retrieved_chunks"),
            "context_tokens": result.get("context_tokens"),
            "reranker_used": result.get("reranker_used"),
            # Evaluation metrics
            "relevance": result.get("relevance"),
            "answer_correct": result.get("answer_correct"),
            # Cost metrics
            "total_tokens": result.get("total_tokens"),
            "cost": result.get("cost"),
        }
    )


@app.post("/feedback")
def feedback():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    conversation_id = data.get("conversation_id")

    try:
        UUID(conversation_id)

    except Exception:
        return jsonify({"error": "Invalid conversation ID"}), 400

    try:
        feedback_value = int(data.get("feedback"))

    except Exception:
        return jsonify({"error": "Feedback must be 1 or -1"}), 400

    if feedback_value not in (1, -1):
        return jsonify({"error": "Feedback must be 1 or -1"}), 400

    try:
        save_feedback(conversation_id, feedback_value)

    except Error:
        return jsonify({"error": "Invalid conversation ID"}), 400

    return jsonify({"message": "Feedback received"})


@app.post("/evaluation")
def evaluation():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    conversation_id = data.get("conversation_id")
    answer_correct = data.get("answer_correct")

    if conversation_id is None or answer_correct is None:
        return jsonify({"error": "Missing evaluation fields"}), 400

    try:
        UUID(conversation_id)

    except Exception:
        return jsonify({"error": "Invalid conversation ID"}), 400

    conn = get_connection()

    try:

        with conn.cursor() as cur:

            cur.execute(
                """
                UPDATE conversations
                SET answer_correct = %s
                WHERE id = %s
                RETURNING id
                """,
                (answer_correct, conversation_id),
            )

            if cur.fetchone() is None:
                conn.rollback()
                return jsonify({"error": "Conversation not found"}), 404

        conn.commit()

    finally:
        conn.close()

    return jsonify({"message": "Evaluation stored"})


@app.get("/health")
def health():

    return jsonify({"status": "ok"})


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
