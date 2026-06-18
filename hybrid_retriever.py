from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, chunks, vector_db):

        self.vector_db = vector_db
        self.chunks = chunks

        self.texts = [
            chunk.page_content
            for chunk in chunks
        ]

        tokenized_corpus = [
            text.lower().split()
            for text in self.texts
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    def search(
        self,
        query,
        vector_k=5,
        bm25_k=5
    ):

        vector_results = (
            self.vector_db.similarity_search(
                query,
                k=vector_k
            )
        )

        tokenized_query = (
            query.lower().split()
        )

        bm25_results = self.bm25.get_top_n(
            tokenized_query,
            self.chunks,
            n=bm25_k
        )

        combined = []

        seen = set()

        for doc in (
            vector_results
            + bm25_results
        ):

            if doc.page_content not in seen:

                combined.append(doc)

                seen.add(
                    doc.page_content
                )

        return combined