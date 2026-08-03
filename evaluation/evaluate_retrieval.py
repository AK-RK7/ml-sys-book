import pandas as pd
from tqdm.auto import tqdm

from study_assistant.ingest import load_index

index = load_index()

def search(query, boost=None, semantic_weight=0.5):

    if boost is None:
        boost = {"text": 0.25}

    return index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=5,
        semantic_weight=semantic_weight,
    )


def hit_rate(relevance_total):
    return sum(any(r) for r in relevance_total) / len(relevance_total)


def mrr(relevance_total):
    score = 0.0

    for rel in relevance_total:
        for rank, ok in enumerate(rel, start=1):
            if ok:
                score += 1 / rank
                break

    return score / len(relevance_total)


def evaluate(ground_truth, search_function):

    relevance_total = []

    for record in tqdm(ground_truth):

        expected = record["chunk_id"]

        results = search_function(
            record["question"]
        )

        relevance = [
            r["chunk_id"] == expected
            for r in results
        ]

        relevance_total.append(relevance)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }

def main():

    df = pd.read_csv(
        "data/ground_truth_retrieval.csv"
    )

    ground_truth = df.to_dict(
        orient="records"
    )

    boost_values = [0.25, 0.30, 0.35, 0.40]
    semantic_weights = [0.60, 0.70, 0.80]

    results = []

    for boost_value in boost_values:

        for semantic_weight in semantic_weights:

            print(
                f"\nTesting "
                f"text_boost={boost_value}, "
                f"semantic_weight={semantic_weight}"
            )

            metrics = evaluate(
                ground_truth,
                lambda q: search(
                    q,
                    boost={"text": boost_value},
                    semantic_weight=semantic_weight,
                )
            )

            results.append(
                {
                    "text_boost": boost_value,
                    "semantic_weight": semantic_weight,
                    "hit_rate": metrics["hit_rate"],
                    "mrr": metrics["mrr"],
                }
            )


    df_results = pd.DataFrame(results)

    print("\nBest results:")
    print(
        df_results.sort_values(
            "mrr",
            ascending=False
        ).head(10)
    )

    df_results.to_csv(
        "data/retrieval_tuning_results.csv",
        index=False,
    )

if __name__ == "__main__":
    main()