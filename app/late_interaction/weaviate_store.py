from __future__ import annotations

import uuid
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter, MetadataQuery

from app.config import settings
from .scoring import validate_dense, validate_multi


class LateInteractionWeaviateStore:
    DENSE_VECTOR = "dense_v4"
    LATE_VECTOR = "late_v4"

    def __init__(self):
        self.client = None

    def connect(self):
        if self.client is not None:
            return self.client
        self.client = weaviate.connect_to_local(
            host=settings.weaviate_host,
            port=settings.weaviate_http_port,
            grpc_port=settings.weaviate_grpc_port,
            additional_config=AdditionalConfig(timeout=Timeout(init=30, query=120, insert=300)),
        )
        if not self.client.is_ready():
            raise RuntimeError("Weaviate is not ready")
        return self.client

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def ensure_collection(self):
        client = self.connect()
        name = settings.li_weaviate_collection
        if client.collections.exists(name):
            return client.collections.use(name)
        return client.collections.create(
            name=name,
            vector_config=[
                Configure.Vectors.self_provided(
                    name=self.DENSE_VECTOR,
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE,
                    ),
                ),
                Configure.MultiVectors.self_provided(name=self.LATE_VECTOR),
            ],
            properties=[
                Property(name="asset_id", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="modality", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="token_count", data_type=DataType.INT),
                Property(name="source_name", data_type=DataType.TEXT),
                Property(name="thumbnail_relpath", data_type=DataType.TEXT),
                Property(name="late_vector_count", data_type=DataType.INT),
                Property(name="embedding_model", data_type=DataType.TEXT),
                Property(name="model_revision", data_type=DataType.TEXT),
                Property(name="dense_dim", data_type=DataType.INT),
                Property(name="late_dim", data_type=DataType.INT),
            ],
        )

    @staticmethod
    def deterministic_uuid(asset_id: str, modality: str, chunk_id: str) -> str:
        key = f"{asset_id}|{modality}|{chunk_id}|jina-v4-late-v1"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def insert(
        self,
        properties: dict[str, Any],
        dense: list[float],
        multi: list[list[float]],
    ) -> str:
        dense = validate_dense(dense)
        multi = validate_multi(multi)
        collection = self.ensure_collection()
        uid = self.deterministic_uuid(
            str(properties["asset_id"]), str(properties["modality"]), str(properties["chunk_id"])
        )
        collection.data.insert(
            properties=properties,
            vector={self.DENSE_VECTOR: dense, self.LATE_VECTOR: multi},
            uuid=uid,
        )
        return uid

    def delete_asset(self, asset_id: str) -> None:
        self.ensure_collection().data.delete_many(
            where=Filter.by_property("asset_id").equal(asset_id)
        )

    def _filters(self, asset_id: str | None, modality: str):
        filters = None
        if asset_id:
            filters = Filter.by_property("asset_id").equal(asset_id)
        if modality != "all":
            mod = Filter.by_property("modality").equal(modality)
            filters = mod if filters is None else (filters & mod)
        return filters

    def _search(
        self,
        vector: Any,
        *,
        target_vector: str,
        asset_id: str | None,
        modality: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        response = self.ensure_collection().query.near_vector(
            near_vector=vector,
            target_vector=target_vector,
            filters=self._filters(asset_id, modality),
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
            return_properties=[
                "asset_id", "chunk_id", "modality", "chunk_index", "text",
                "token_count", "source_name", "thumbnail_relpath", "late_vector_count",
                "embedding_model", "model_revision", "dense_dim", "late_dim",
            ],
        )
        return [
            {"uuid": str(obj.uuid), **obj.properties, "distance": obj.metadata.distance}
            for obj in response.objects
        ]

    def search_dense(
        self, vector: list[float], asset_id: str | None, modality: str, limit: int
    ) -> list[dict[str, Any]]:
        return self._search(
            validate_dense(vector),
            target_vector=self.DENSE_VECTOR,
            asset_id=asset_id,
            modality=modality,
            limit=limit,
        )

    def search_late(
        self, vectors: list[list[float]], asset_id: str | None, modality: str, limit: int
    ) -> list[dict[str, Any]]:
        return self._search(
            validate_multi(vectors),
            target_vector=self.LATE_VECTOR,
            asset_id=asset_id,
            modality=modality,
            limit=limit,
        )

    def count_asset(self, asset_id: str) -> int:
        response = self.ensure_collection().aggregate.over_all(
            filters=Filter.by_property("asset_id").equal(asset_id), total_count=True
        )
        return int(response.total_count or 0)


store = LateInteractionWeaviateStore()
