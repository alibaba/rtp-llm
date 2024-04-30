import logging.config
from typing import Any, Dict

from fastapi import FastAPI
from fastapi import Request as RawRequest
from maga_transformer.server.inference_server import InferenceServer
from maga_transformer.server.misc import check_is_master


def register_embedding_api(app: FastAPI, inference_server: InferenceServer):
    # entry for worker RANK == 0
    @app.post("/v1/embeddings")    
    @check_is_master()
    async def embedding(request: Dict[str, Any], raw_request: RawRequest):
        return await inference_server.embedding(request, raw_request)
    
    @app.post("/v1/embeddings/similarity")
    @check_is_master()
    async def similarity(request: Dict[str, Any], raw_request: RawRequest):
        return await inference_server.similarity(request, raw_request)

    @app.post("/v1/classifier")
    @check_is_master()
    async def classifier(request: Dict[str, Any], raw_request: RawRequest):
        return await inference_server.classifier(request, raw_request)