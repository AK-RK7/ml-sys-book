import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer


class Index:
    """
    Hybrid retrieval index using:
      - TF-IDF lexical search
      - SentenceTransformer semantic search

    Supports:
      - Field boosting
      - Metadata filtering
      - Hybrid score weighting
    """

    def __init__(
        self,
        text_fields,
        keyword_fields=None,
        vectorizer_params=None,
        semantic_model="all-MiniLM-L6-v2",
    ):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields or []

        vectorizer_params = vectorizer_params or {}

        self.vectorizers = {
            field: TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=50000,
                **vectorizer_params,
            )
            for field in self.text_fields
        }

        self.text_matrices = {}

        self.semantic_model = SentenceTransformer(
            semantic_model
        )

        self.semantic_embeddings = None

        self.keyword_df = None
        self.documents = []

    def fit(self, documents):
        """
        Build TF-IDF indices and semantic embeddings.
        """

        self.documents = documents

        keyword_data = {
            field: []
            for field in self.keyword_fields
        }

        # Build TF-IDF matrices for every searchable field
        for field in self.text_fields:
            texts = [
                str(doc.get(field, ""))
                for doc in documents
            ]

            self.text_matrices[field] = (
                self.vectorizers[field]
                .fit_transform(texts)
            )

        # One semantic embedding per document
        combined_texts = [
            " ".join(
                str(doc.get(field, ""))
                for field in self.text_fields
            )
            for doc in documents
        ]

        self.semantic_embeddings = (
            self.semantic_model.encode(
                combined_texts,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        )

        for doc in documents:
            for field in self.keyword_fields:
                keyword_data[field].append(
                    doc.get(field, "")
                )

        self.keyword_df = pd.DataFrame(keyword_data)

        return self

    def search(
        self,
        query,
        filter_dict=None,
        boost_dict=0.25,
        num_results=5,
        semantic_weight=0.60,
    ):
        """
        Hybrid search.

        Final score =
            lexical score
            + semantic_weight × semantic similarity
        """

        filter_dict = filter_dict or {}
        boost_dict = boost_dict or {}

        scores = np.zeros(len(self.documents))

        #
        # TF-IDF lexical search
        #
        for field in self.text_fields:

            query_vector = (
                self.vectorizers[field]
                .transform([query])
            )

            tfidf_score = (
                cosine_similarity(
                    query_vector,
                    self.text_matrices[field],
                )
                .flatten()
            )

            # Normalize so one field doesn't dominate
            if tfidf_score.max() > 0:
                tfidf_score = (
                    tfidf_score /
                    tfidf_score.max()
                )

            boost = boost_dict.get(field, 1.0)

            scores += boost * tfidf_score

        #
        # Semantic search
        #
        if semantic_weight > 0:

            query_embedding = (
                self.semantic_model.encode(
                    [query],
                    normalize_embeddings=True,
                )
            )

            semantic_score = (
                cosine_similarity(
                    query_embedding,
                    self.semantic_embeddings,
                )
                .flatten()
            )

            scores += semantic_weight * semantic_score

        #
        # Metadata filtering
        #
        for field, value in filter_dict.items():

            if field not in self.keyword_fields:
                continue

            mask = (
                self.keyword_df[field]
                == value
            ).to_numpy()

            scores *= mask

        #
        # Ranking
        #
        if len(scores) == 0:
            return []

        top_indices = np.argsort(
            -scores
        )[:num_results]

        results = []

        for idx in top_indices:

            if scores[idx] <= 0:
                continue

            result = dict(self.documents[idx])
            result["_score"] = float(scores[idx])

            results.append(result)

        return results