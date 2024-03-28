import os
import sys
import json
import time
import logging
import logging.config
import uvicorn
import traceback
from typing import Generator, Union, Any, Dict, List, AsyncGenerator, Callable, Coroutine

from fastapi import FastAPI
from fastapi import Request as RawRequest
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter

from maga_transformer.distribute.worker_info import g_worker_info, g_parallel_info
from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.openai.api_datatype import ChatCompletionRequest, ChatCompletionStreamResponse
from maga_transformer.embedding.api_datatype import OpenAIEmbeddingRequest, OpenAIEmbeddingResponse, SimilarityRequest
from maga_transformer.utils.version_info import VersionInfo
from maga_transformer.config.uvicorn_config import UVICORN_LOGGING_CONFIG
from maga_transformer.models.base_model import BaseModel
from maga_transformer.server.inference_server import InferenceServer
from maga_transformer.config.exceptions import ExceptionType


def register_embedding_api(app: FastAPI, inference_server: InferenceServer):
    # entry for worker RANK == 0
    @app.post("/v1/embeddings")
    async def embedding(request: OpenAIEmbeddingRequest, raw_request: RawRequest):
        if not g_parallel_info.is_master:
            return InferenceServer.format_exception(ExceptionType.UNSUPPORTED_OPERATION,
                                "gang worker should not access this completions api directly!")
        return await inference_server.embedding(request, raw_request)
    
    @app.post("/v1/embeddings/similarity")
    async def similarity(request: SimilarityRequest, raw_request: RawRequest):
        if not g_parallel_info.is_master:
            return InferenceServer.format_exception(ExceptionType.UNSUPPORTED_OPERATION,
                                "gang worker should not access this completions api directly!")
        return await inference_server.similarity(request, raw_request)
