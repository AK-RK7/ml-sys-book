import json
import pandas as pd
from tqdm.auto import tqdm

from study_assistant import rag as rag_module


def main():

    df = pd.read_csv(
        "data/ground_truth_retrieval.csv"
    )

    df_sample = df.sample(
        n=200,
        random_state=1
    )

    results = []


    for _, row in tqdm(
        df_sample.iterrows(),
        total=len(df_sample)
    ):

        question = row["question"]

        output = rag_module.rag(question)


        results.append(
            {
                "doc_id": row["doc_id"],
                "chunk_id": row["chunk_id"],
                "question": question,

                "answer": output["answer"],

                "sources": output["sources"],

                "retrieved_documents":
                    output["retrieved_documents"],

                "relevance":
                    output["relevance"],

                "relevance_explanation":
                    output["relevance_explanation"],

                "response_time":
                    output["response_time"],

                "model":
                    output["model_used"],

                "total_tokens":
                    output["total_tokens"],

                "cost":
                    output["openai_cost"],
            }
        )


    df_results = pd.DataFrame(results)


    print("\nRelevance distribution:")

    print(
        df_results["relevance"]
        .value_counts(normalize=True)
    )


    df_results.to_csv(
        "data/rag_evaluation_results.csv",
        index=False,
    )


    print(
        "\nSaved:",
        "data/rag_evaluation_results.csv"
    )


if __name__ == "__main__":
    main()