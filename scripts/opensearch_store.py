from typing import List, Tuple, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection
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
    ):
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
        self.index_name = index_name
        self.embedding_dim = embedding_dim

    def ensure_index(self) -> None:
        if self.client.indices.exists(self.index_name):
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
                            "engine": "nmslib",
                        },
                    },
                }
            },
        }
        self.client.indices.create(index=self.index_name, body=body)

    def index_documents(self, df, embeddings: np.ndarray, batch_size: int = 1000) -> None:
        actions = []
        for i, row in df.iterrows():
            doc = {
                "title": row["title"],
                "title_with_year": row["title_with_year"],
                "tags": row.get("tags", ""),
                "embedding": embeddings[i].tolist(),
            }
            actions.append({"index": {"_index": self.index_name, "_id": int(i)}})
            actions.append(doc)
            if len(actions) >= 2 * batch_size:
                self.client.bulk(body=actions)
                actions = []
        if actions:
            self.client.bulk(body=actions)
        self.client.indices.refresh(index=self.index_name)

    def knn_query_by_id(self, id_: int, k: int = 10) -> List[Tuple[int, float]]:
        src = self.client.get(index=self.index_name, id=int(id_))["_source"]
        vec = src["embedding"]
        return self.knn_query(vec, k=k)

    def knn_query(self, vector: List[float], k: int = 10) -> List[Tuple[int, float]]:
        body = {
            "size": k + 1,
            "query": {"knn": {"embedding": {"vector": vector, "k": k + 1}}},
            "_source": False,
        }
        res = self.client.search(index=self.index_name, body=body)
        results = []
        for hit in res["hits"]["hits"]:
            results.append((int(hit["_id"]), float(hit["_score"])))
        return results
