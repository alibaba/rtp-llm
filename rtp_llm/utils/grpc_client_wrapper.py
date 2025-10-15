import asyncio
import logging
from typing import Any, Dict, Optional
import grpc
from rtp_llm.cpp.proto.model_rpc_service_pb2 import (
    BroadcastLoadRequestPB,
    BroadcastLoadResponsePB,
    CacheStatusPB,
    CacheVersionPB,
    EmptyPB,
    GenerateOutputsPB,
    GenerateRequestPB,
    MMPreprocessConfigPB,
    MultimodalInputPB,
    MultimodalInputsPB,
    MultimodalOutputsPB,
    RemoteFinishRequestPB,
    StatusVersionPB,
    TensorPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceStub,
    RpcServiceStub,
)
class GrpcClientWrapper:
    """Wrapper for direct gRPC calls to replace async_request_server"""
    def __init__(self, server_port: int):
        self.server_port = server_port
        self.address = f"localhost:{server_port}"
        self.channel = None
        self.stub = None
        self.multimodal_stub = None
    async def _ensure_connection(self):
        """Ensure gRPC channel and stub are created"""
        if self.channel is None or self.stub is None:
            self.channel = grpc.aio.insecure_channel(
                self.address,
                options=[
                    ("grpc.max_metadata_size", 1024 * 1024 * 1024),
                ],
            )
            self.stub = RpcServiceStub(self.channel)
            self.multimodal_stub = MultimodalRpcServiceStub(self.channel)
    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None
    async def health_check(self) -> str:
        """Check server health"""
        try:
            await self._ensure_connection()
            # Using a simple request to check if server is responsive
            request = StatusVersionPB()
            request.latest_cache_version = -1
            request.latest_finished_version = -1
            # response = await self.stub.GetWorkerStatus(request, timeout=5)
            return "ok"
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return "error"
    async def get_cache_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get cache status from gRPC server"""
        try:
            await self._ensure_connection()
            request = CacheVersionPB()
            request.latest_cache_version = int(
                query_params.get("latest_cache_version", -1)
            )
            response = await self.stub.GetCacheStatus(request, timeout=10)
            # Convert response to dict format expected by frontend
            result = {
                "available_kv_cache": response.available_kv_cache,
                "total_kv_cache": response.total_kv_cache,
                "block_size": response.block_size,
                "version": response.version,
            }
            # Add cache_keys if they exist
            if hasattr(response, "cache_keys"):
                result["cached_keys"] = list(response.cache_keys)
            # Add frontend_available_concurrency as it was in the original implementation
            result["frontend_available_concurrency"] = (
                0  # This would need to be set properly
            )
            return result
        except Exception as e:
            logging.error(f"Get cache status failed: {e}")
            return {"error": f"Failed to get cache status: {str(e)}"}
    async def get_worker_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get worker status from gRPC server"""
        try:
            await self._ensure_connection()
            request = StatusVersionPB()
            request.latest_cache_version = int(
                query_params.get("latest_cache_version", -1)
            )
            request.latest_finished_version = int(
                query_params.get("latest_finised_version", -1)
            )  # Typo in original code
            response = await self.stub.GetWorkerStatus(request, timeout=10)
            # Convert response to dict format expected by frontend
            result = {
                "role": response.role,
                "available_concurrency": response.available_concurrency,
                "waiting_query_len": response.waiting_query_len,
                "running_query_len": response.running_query_len,
                "step_latency_ms": response.step_latency_ms,
                "iterate_count": response.iterate_count,
                "dp_size": response.dp_size,
                "tp_size": response.tp_size,
                "version": response.version,
                "status_version": response.status_version,
                "alive": response.alive,
                "precision": response.precision,
            }
            # Add frontend_available_concurrency as it was in the original implementation
            result["frontend_available_concurrency"] = (
                0  # This would need to be set properly
            )
            return result
        except Exception as e:
            logging.error(f"Get worker status failed: {e}")
            return {"error": f"Failed to get worker status: {str(e)}"}
    async def update_model(self, version_info: Dict[str, Any]) -> Dict[str, Any]:
        """Update model/version info - this would need to be implemented based on your requirements"""
        try:
            # This is a placeholder - would need specific implementation
            return {
                "status": "ok",
                "message": "Update not implemented in direct gRPC mode",
            }
        except Exception as e:
            logging.error(f"Update model failed: {e}")
            return {"error": f"Failed to update model: {str(e)}"}
    async def set_log_level(self, req: Any) -> Dict[str, Any]:
        """Set log level - this would need to be implemented based on your requirements"""
        try:
            # This is a placeholder - would need specific implementation
            return {
                "status": "ok",
                "message": "Set log level not implemented in direct gRPC mode",
            }
        except Exception as e:
            logging.error(f"Set log level failed: {e}")
            return {"error": f"Failed to set log level: {str(e)}"}
    async def update_eplb_config(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Update EPLB config - this would need to be implemented based on your requirements"""
        try:
            # This is a placeholder - would need specific implementation
            return {
                "status": "ok",
                "message": "Update EPLB config not implemented in direct gRPC mode",
            }
        except Exception as e:
            logging.error(f"Update EPLB config failed: {e}")
            return {"error": f"Failed to update EPLB config: {str(e)}"}
    async def update_scheduler_info(self, req: Any) -> Dict[str, Any]:
        """Update scheduler info - this would need to be implemented based on your requirements"""
        try:
            # This is a placeholder - would need specific implementation
            return {
                "status": "ok",
                "message": "Update scheduler info not implemented in direct gRPC mode",
            }
        except Exception as e:
            logging.error(f"Update scheduler info failed: {e}")
            return {"error": f"Failed to update scheduler info: {str(e)}"}
    async def post_request(self, uri: str, req: Dict[str, Any]) -> Dict[str, Any]:
        """Generic POST request handler - routes to appropriate method based on URI"""
        try:
            if uri == "health_check":
                return await self.health_check()
            elif uri == "cache_status":
                return await self.get_cache_status(req)
            elif uri == "worker_status":
                return await self.get_worker_status(req)
            elif uri == "update":
                return await self.update_model(req)
            elif uri == "set_log_level":
                return await self.set_log_level(req)
            elif uri == "update_eplb_config":
                return await self.update_eplb_config(req)
            elif uri == "update_scheduler_info":
                return await self.update_scheduler_info(req)
            elif uri == "v1/embeddings":
                # Handle embedding requests
                return await self.handle_embedding_request(req)
            else:
                # Default case - return empty success
                return {"status": "ok"}
        except Exception as e:
            logging.error(f"POST request to {uri} failed: {e}")
            return {"error": f"Request failed: {str(e)}"}
    async def get_request(self, uri: str, req: Dict[str, Any]) -> Dict[str, Any]:
        """Generic GET request handler - routes to appropriate method based on URI"""
        try:
            if uri == "" or uri == "/":  # Root endpoint
                # Check health for root endpoint
                health = await self.health_check()
                if health == "ok":
                    return {"status": "home"}
                else:
                    return {"error": "Server not ready"}
            elif uri == "health" or uri == "health_check":
                health = await self.health_check()
                return health
            elif uri == "cache_status":
                return await self.get_cache_status(req)
            elif uri == "worker_status":
                return await self.get_worker_status(req)
            else:
                # Default case - return empty success
                return {"status": "ok"}
        except Exception as e:
            logging.error(f"GET request to {uri} failed: {e}")
            return {"error": f"Request failed: {str(e)}"}
    async def handle_embedding_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Handle embedding requests by making gRPC calls to the multimodal service"""
        try:
            await self._ensure_connection()
            # Convert request to MultimodalInputsPB
            multimodal_inputs = MultimodalInputsPB()
            # Handle different types of embedding requests
            if isinstance(req, dict):
                # Handle various embedding request formats
                if "input" in req:
                    # Standard OpenAI embedding format
                    inputs = req["input"]
                    if isinstance(inputs, str):
                        inputs = [inputs]
                    elif not isinstance(inputs, list):
                        inputs = [str(inputs)]
                    for input_item in inputs:
                        multimodal_input = MultimodalInputPB()
                        # For text embeddings, we treat the text as a URL-like input
                        multimodal_input.multimodal_url = input_item
                        multimodal_input.multimodal_type = 0  # Text input type
                        multimodal_inputs.multimodal_inputs.append(multimodal_input)
                elif "texts" in req:
                    # Custom format with texts array
                    texts = req["texts"]
                    if isinstance(texts, str):
                        texts = [texts]
                    for text in texts:
                        multimodal_input = MultimodalInputPB()
                        multimodal_input.multimodal_url = text
                        multimodal_input.multimodal_type = 0  # Text input type
                        multimodal_inputs.multimodal_inputs.append(multimodal_input)
                else:
                    # Handle generic dict format
                    multimodal_input = MultimodalInputPB()
                    # Convert the entire request to a string representation
                    multimodal_input.multimodal_url = str(req)
                    multimodal_input.multimodal_type = 0  # Text input type
                    multimodal_inputs.multimodal_inputs.append(multimodal_input)
            # Make the gRPC call to the multimodal service
            response = await self.multimodal_stub.RemoteMultimodalEmbedding(
                multimodal_inputs, timeout=30
            )
            # Convert response to dict format
            result = {
                "data": [],
                "model": "rtp-llm-embedding",
                "object": "list",
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
            # Process the multimodal outputs
            for i, multimodal_output in enumerate(response.multimodal_outputs):
                # Convert tensor data to embedding format
                if multimodal_output.HasField("multimodal_embedding"):
                    embedding_tensor = multimodal_output.multimodal_embedding
                    # Extract embedding data based on tensor type
                    if (
                        embedding_tensor.data_type == TensorPB.DataType.FP32
                        and embedding_tensor.fp32_data
                    ):
                        # Handle FP32 data
                        import struct
                        float_count = len(embedding_tensor.fp32_data) // 4
                        embedding_data = list(
                            struct.unpack(f"{float_count}f", embedding_tensor.fp32_data)
                        )
                    elif (
                        embedding_tensor.data_type == TensorPB.DataType.FP16
                        and embedding_tensor.fp16_data
                    ):
                        # Handle FP16 data (convert to FP32 for compatibility)
                        import struct
                        half_count = len(embedding_tensor.fp16_data) // 2
                        # Simple FP16 to FP32 conversion (this is a basic implementation)
                        half_data = struct.unpack(
                            f"{half_count}H", embedding_tensor.fp16_data
                        )
                        embedding_data = [
                            self._convert_fp16_to_fp32(h) for h in half_data
                        ]
                    else:
                        # Fallback to empty embedding
                        embedding_data = []
                    result["data"].append(
                        {"embedding": embedding_data, "index": i, "object": "embedding"}
                    )
            return result
        except grpc.aio.AioRpcError as e:
            logging.error(f"Embedding request gRPC error: {e}")
            return {"error": f"gRPC error: {e.details()}", "code": e.code().name}
        except Exception as e:
            logging.error(f"Embedding request failed: {e}")
            return {"error": f"Embedding request failed: {str(e)}"}
    def _convert_fp16_to_fp32(self, half: int) -> float:
        """Convert FP16 bit representation to FP32 float"""
        # Simple implementation - in practice, you might want to use numpy or struct.pack/unpack
        # Extract sign, exponent, and mantissa
        sign = (half >> 15) & 0x1
        exp = (half >> 10) & 0x1F
        mant = half & 0x3FF
        if exp == 0:
            if mant == 0:
                return -0.0 if sign else 0.0
            else:
                # Subnormal number
                return (-1) ** sign * 2 ** (-14) * (mant / 1024.0)
        elif exp == 31:
            if mant == 0:
                return float("-inf") if sign else float("inf")
            else:
                return float("nan")
        else:
            # Normal number
            return (-1) ** sign * 2 ** (exp - 15) * (1 + mant / 1024.0)