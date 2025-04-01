import logging.config
from typing import Any, Dict

from fastapi import FastAPI
from fastapi import Request as RawRequest
from maga_transformer.embedding.embedding_type import TYPE_STR, EmbeddingType
from maga_transformer.utils.util import request_server

def register_frontend_embedding_api(app: FastAPI, backend_server_port: int):
    # 通过路径区别请求的方式，为后续可能存在的多种task类型做后向兼容
    @app.post("/v1/embeddings")
    async def embedding(request: Dict[str, Any], raw_request: RawRequest):
        return request_server("post", backend_server_port, "v1/embeddings", request)

    @app.post("/v1/embeddings/dense")
    async def embedding_dense(request: Dict[str, Any], raw_request: RawRequest):
        return request_server("post", backend_server_port, "v1/embeddings/dense", request)

    @app.post("/v1/embeddings/sparse")
    async def embedding_sparse(request: Dict[str, Any], raw_request: RawRequest):
        return request_server("post", backend_server_port, "v1/embeddings/sparse", request)


    @app.post("/v1/embeddings/colbert")
    async def embedding_colbert(request: Dict[str, Any], raw_request: RawRequest):
        return request_server("post", backend_server_port, "v1/embeddings/colbert", request)

    @app.post("/v1/embeddings/similarity")
    async def similarity(request: Dict[str, Any], raw_request: RawRequest):
        return request_server("post", backend_server_port, "v1/embeddings/similarity", request)

    @app.post("/v1/classifier")
    async def classifier(request: Dict[str, Any], raw_request: RawRequest):
        return request_server("post", backend_server_port, "v1/classifier", request)

    @app.post("/v1/reranker")
    async def reranker(request: Dict[str, Any], raw_request: RawRequest):
        return request_server("post", backend_server_port, "v1/reranker", request)