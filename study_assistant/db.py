import os
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import DictCursor


def get_connection():

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB", "study_assistant"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


def init_db():

    conn = get_connection()

    try:

        with conn.cursor() as cur:

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (

                    id UUID PRIMARY KEY,

                    question TEXT NOT NULL,

                    rewritten_query TEXT,

                    answer TEXT NOT NULL,

                    model_used TEXT NOT NULL,

                    response_time DOUBLE PRECISION NOT NULL,

                    prompt_version TEXT,

                    retrieval_method TEXT DEFAULT 'hybrid',

                    reranker_used BOOLEAN DEFAULT FALSE,

                    retrieved_chunks INTEGER DEFAULT 0,

                    context_tokens INTEGER DEFAULT 0,

                    top_retrieval_score DOUBLE PRECISION,

                    avg_retrieval_score DOUBLE PRECISION,

                    relevance TEXT,

                    answer_correct BOOLEAN,

                    total_tokens INTEGER DEFAULT 0,

                    cost DOUBLE PRECISION DEFAULT 0,

                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP

                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (

                    id SERIAL PRIMARY KEY,

                    conversation_id UUID
                        REFERENCES conversations(id)
                        ON DELETE CASCADE,

                    feedback INTEGER NOT NULL
                        CHECK(feedback IN (-1,1)),

                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP

                );
                """
            )

            # migrations for old databases

            cur.execute(
                """
                ALTER TABLE conversations
                ADD COLUMN IF NOT EXISTS rewritten_query TEXT;
                """
            )

            cur.execute(
                """
                ALTER TABLE conversations
                ADD COLUMN IF NOT EXISTS top_retrieval_score DOUBLE PRECISION;
                """
            )

            cur.execute(
                """
                ALTER TABLE conversations
                ADD COLUMN IF NOT EXISTS avg_retrieval_score DOUBLE PRECISION;
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
                ON conversations(timestamp);
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_model
                ON conversations(model_used);
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_retrieval_method
                ON conversations(retrieval_method);
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_answer_correct
                ON conversations(answer_correct);
                """
            )

        conn.commit()

    finally:

        conn.close()


def save_conversation(
    conversation_id,
    question,
    answer_data,
    timestamp=None,
):

    if timestamp is None:

        timestamp = datetime.now(timezone.utc)

    conn = get_connection()

    try:

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO conversations
                (

                    id,

                    question,

                    rewritten_query,

                    answer,

                    model_used,

                    response_time,

                    prompt_version,

                    retrieval_method,

                    reranker_used,

                    retrieved_chunks,

                    context_tokens,

                    top_retrieval_score,

                    avg_retrieval_score,

                    relevance,

                    answer_correct,

                    total_tokens,

                    cost,

                    timestamp

                )

                VALUES

                (

                    %s,%s,%s,

                    %s,

                    %s,%s,%s,

                    %s,%s,%s,%s,

                    %s,%s,

                    %s,%s,

                    %s,%s,

                    %s

                )

                """,
                (
                    str(conversation_id),
                    question,
                    answer_data.get("rewritten_query", question),
                    answer_data["answer"],
                    answer_data.get("model_used", "unknown"),
                    answer_data.get("response_time", 0),
                    answer_data.get("prompt_version"),
                    answer_data.get("retrieval_method", "hybrid"),
                    answer_data.get("reranker_used", False),
                    answer_data.get("retrieved_chunks", 0),
                    answer_data.get("context_tokens", 0),
                    answer_data.get("top_retrieval_score"),
                    answer_data.get("avg_retrieval_score"),
                    answer_data.get("relevance"),
                    answer_data.get("answer_correct"),
                    answer_data.get("total_tokens", 0),
                    answer_data.get("cost", 0.0),
                    timestamp,
                ),
            )

        conn.commit()

    finally:

        conn.close()


def save_feedback(conversation_id, feedback):

    conn = get_connection()

    try:

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO feedback
                (
                    conversation_id,
                    feedback,
                    timestamp
                )

                VALUES
                (%s,%s,%s)

                """,
                (
                    str(conversation_id),
                    feedback,
                    datetime.now(timezone.utc),
                ),
            )

        conn.commit()

    finally:

        conn.close()


def get_recent_conversations(limit=10):

    conn = get_connection()

    try:

        with conn.cursor(cursor_factory=DictCursor) as cur:

            cur.execute(
                """
                SELECT *

                FROM conversations

                ORDER BY timestamp DESC

                LIMIT %s

                """,
                (limit,),
            )

            return cur.fetchall()

    finally:

        conn.close()


def get_feedback_stats():

    conn = get_connection()

    try:

        with conn.cursor(cursor_factory=DictCursor) as cur:

            cur.execute(
                """
                SELECT

                    SUM(
                        CASE
                            WHEN feedback = 1
                            THEN 1
                            ELSE 0
                        END
                    ) AS thumbs_up,


                    SUM(
                        CASE
                            WHEN feedback = -1
                            THEN 1
                            ELSE 0
                        END
                    ) AS thumbs_down


                FROM feedback

                """
            )

            return cur.fetchone()

    finally:

        conn.close()
