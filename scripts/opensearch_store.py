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
                "_meta": {
                    "dataset_hash": "",
                    "embedding_model": "",
                    "embedding_dim": self.embedding_dim
                },
                "properties": {
                    "movie_id": {"type": "long"},
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

    def get_index_meta(self) -> Dict[str, Any]:
        """Get index metadata."""
        m = self.client.indices.get_mapping(index=self.index_name)
        return m[self.index_name]["mappings"].get("_meta", {})

    def put_index_meta(self, **kv) -> None:
        """Update index metadata."""
        self.client.indices.put_mapping(index=self.index_name, body={"_meta": kv})

    def get_doc_count(self) -> int:
        """Get document count in index."""
        s = self.client.cat.count(index=self.index_name, format="json")
        return int(s[0]["count"])

    def _doc_actions(self, df, embeddings: np.ndarray) -> Iterator[Dict[str, Any]]:
        """Generate document actions for bulk indexing with stable movie_id."""
        mid = df["movie_id"].to_numpy()
        titles = df["title"].to_numpy()
        twy = df["title_with_year"].to_numpy()
        tags = df["tags"].to_numpy()
        for i in range(len(df)):
            yield {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": int(mid[i]),
                "_source": {
                    "movie_id": int(mid[i]),
                    "title": titles[i],
                    "title_with_year": twy[i],
                    "tags": tags[i] if isinstance(tags[i], str) else "",
                    "embedding": embeddings[i].tolist(),
                },
            }

    def index_documents(self, df, embeddings: np.ndarray, chunk_size: int = 1000, request_timeout: int = 60) -> None:
        """Index documents (legacy method, use index_documents_if_needed instead)."""
        bulk(self.client, self._doc_actions(df, embeddings), chunk_size=chunk_size, request_timeout=request_timeout)
        self.client.indices.refresh(index=self.index_name)

    def index_documents_if_needed(
        self,
        df,
        embeddings: np.ndarray,
        dataset_hash: str,
        model_name: str,
        chunk_size: int = 2000
    ) -> bool:
        """
        Index documents only if needed (idempotent operation).
        
        Returns True if indexing was performed, False if skipped.
        """
        meta = self.get_index_meta()
        up_to_date = (
            meta.get("dataset_hash") == dataset_hash
            and meta.get("embedding_model") == model_name
            and int(meta.get("embedding_dim", self.embedding_dim)) == self.embedding_dim
            and self.get_doc_count() == len(df)
        )
        
        if up_to_date:
            return False
        
        from opensearchpy.helpers import parallel_bulk
        for ok, _ in parallel_bulk(
            self.client,
            self._doc_actions(df, embeddings),
            thread_count=4,
            chunk_size=chunk_size,
            request_timeout=120
        ):
            pass
        
        self.client.indices.refresh(index=self.index_name)
        self.put_index_meta(
            dataset_hash=dataset_hash,
            embedding_model=model_name,
            embedding_dim=self.embedding_dim
        )
        return True

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
