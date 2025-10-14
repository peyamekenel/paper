from typing import List, Tuple, Optional, Iterator, Dict, Any
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
import numpy as np


class OpenSearchVectorStore:
    def __init__(
        self,
        host: str,
        port: int = 443,
        use_ssl: bool = True,
        http_auth: Optional[Tuple[str, str]] = None,
        index_name: str = "movies",
        embedding_dim: int = 384,
        timeout: int = 30,
        max_retries: int = 3,
        retry_on_timeout: bool = True,
    ):
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=timeout,
            max_retries=max_retries,
            retry_on_timeout=retry_on_timeout,
        )
        self.index_name = index_name
        self.embedding_dim = embedding_dim

    def ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            return
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "title": {"type": "keyword"},
                    "title_with_year": {"type": "keyword"},
                    "tags": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss",
                        },
                    },
                }
            },
        }
        self.client.indices.create(index=self.index_name, body=body)

    def _doc_actions(self, df, embeddings: np.ndarray) -> Iterator[Dict[str, Any]]:
        for i, row in enumerate(df.itertuples(index=False)):
            yield {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": int(i),
                "_source": {
                    "title": getattr(row, "title"),
                    "title_with_year": getattr(row, "title_with_year"),
                    "tags": getattr(row, "tags", ""),
                    "embedding": embeddings[i].tolist(),
                },
            }

    def index_documents(self, df, embeddings: np.ndarray, chunk_size: int = 1000, request_timeout: int = 60) -> None:
        bulk(self.client, self._doc_actions(df, embeddings), chunk_size=chunk_size, request_timeout=request_timeout)
        self.client.indices.refresh(index=self.index_name)

    def knn_query_by_id(self, id_: int, k: int = 10) -> List[Tuple[int, float]]:
        src = self.client.get(index=self.index_name, id=int(id_))["_source"]
        vec = src["embedding"]
        results = self.knn_query(vec, k=k + 1)
        filtered = [(i, s) for (i, s) in results if i != int(id_)]
        return filtered[:k]

    def knn_query(self, vector: List[float], k: int = 10) -> List[Tuple[int, float]]:
        body = {
            "size": k,
            "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
            "_source": False,
        }
        res = self.client.search(index=self.index_name, body=body)
        results = []
        for hit in res["hits"]["hits"]:
            results.append((int(hit["_id"]), float(hit["_score"])))
        return results
