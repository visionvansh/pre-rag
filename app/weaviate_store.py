from __future__ import annotations

import uuid
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter, MetadataQuery

from .config import settings


class WeaviateStore:
    def __init__(self):
        self.client = None

    def connect(self):
        if self.client is not None:
            return self.client
        self.client = weaviate.connect_to_local(
            host=settings.weaviate_host,
            port=settings.weaviate_http_port,
            grpc_port=settings.weaviate_grpc_port,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=180)
            ),
        )
        if not self.client.is_ready():
            raise RuntimeError("Weaviate is not ready")
        return self.client

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    def ensure_collection(self):
        client = self.connect()
        if client.collections.exists(settings.weaviate_collection):
            return client.collections.use(settings.weaviate_collection)

        return client.collections.create(
            name=settings.weaviate_collection,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                )
            ),
            properties=[
                Property(name="asset_id", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="modality", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="start_sec", data_type=DataType.NUMBER),
                Property(name="end_sec", data_type=DataType.NUMBER),
                Property(name="duration_sec", data_type=DataType.NUMBER),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="speaker_ids", data_type=DataType.INT_ARRAY),
                Property(name="token_count", data_type=DataType.INT),
                Property(name="frame_count", data_type=DataType.INT),
                Property(name="source_name", data_type=DataType.TEXT),
                Property(name="thumbnail_relpath", data_type=DataType.TEXT),
                Property(name="embedding_model", data_type=DataType.TEXT),
                Property(name="embedding_dim", data_type=DataType.INT),
            ],
        )

    @staticmethod
    def deterministic_uuid(asset_id: str, modality: str, chunk_id: str) -> str:
        key = f"{asset_id}|{modality}|{chunk_id}|jina-v5-omni-small-1024"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def insert(self, properties: dict[str, Any], vector: list[float]) -> str:
        collection = self.ensure_collection()
        uid = self.deterministic_uuid(
            properties["asset_id"], properties["modality"], properties["chunk_id"]
        )
        collection.data.insert(properties=properties, vector=vector, uuid=uid)
        return uid

    def delete_asset(self, asset_id: str) -> None:
        collection = self.ensure_collection()
        collection.data.delete_many(
            where=Filter.by_property("asset_id").equal(asset_id)
        )

    def search(
        self,
        query_vector: list[float],
        asset_id: str | None,
        modality: str,
        limit: int,
    ) -> list[dict]:
        collection = self.ensure_collection()
        filters = None
        if asset_id:
            filters = Filter.by_property("asset_id").equal(asset_id)
        if modality != "all":
            mod_filter = Filter.by_property("modality").equal(modality)
            filters = mod_filter if filters is None else (filters & mod_filter)

        response = collection.query.near_vector(
            near_vector=query_vector,
            filters=filters,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
            return_properties=[
                "asset_id", "chunk_id", "modality", "chunk_index",
                "start_sec", "end_sec", "text", "speaker_ids",
                "thumbnail_relpath",
            ],
        )
        rows = []
        for obj in response.objects:
            rows.append({
                "uuid": str(obj.uuid),
                **obj.properties,
                "distance": obj.metadata.distance,
            })
        return rows

    def count_asset(self, asset_id: str) -> int:
        collection = self.ensure_collection()
        response = collection.aggregate.over_all(
            filters=Filter.by_property("asset_id").equal(asset_id),
            total_count=True,
        )
        return int(response.total_count or 0)


store = WeaviateStore()
