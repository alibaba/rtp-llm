import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

import grpc
from google.protobuf.json_format import MessageToDict

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.utils.time_util import Timer


class GrpcClientWrapper:
    """Wrapper for direct gRPC calls to replace async_request_server"""

    def __init__(self, server_port: int):
        self.server_port = server_port
        self.address = f"localhost:{server_port}"
        self.channel = None
        self.stub = None

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

    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None

    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        try:
            await self._ensure_connection()
            # Using a simple request to check if server is responsive
            request = pb2.EmptyPB()
            await self.stub.CheckHealth(request, timeout=1)
            return {"status": "ok"}
        except Exception as e:
            return {
                "status": "error",
                "message": e,
            }

    async def get_cache_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get cache status from gRPC server"""
        try:
            start_time = time.time() * 1000
            await self._ensure_connection()
            request = pb2.CacheVersionPB(
                latest_cache_version=query_params.get("latest_cache_version", -1),
                need_cache_keys=query_params.get("need_cache_keys", True),
            )
            response = await self.stub.GetCacheStatus(request, timeout=1)
            # Convert response to dict format expected by frontend
            result = MessageToDict(
                response,
                preserving_proto_field_name=True,
                including_default_value_fields=True,
            )
            kmonitor.report(AccMetrics.CACHE_STATUS_QPS_METRIC, 1)
            kmonitor.report(
                GaugeMetrics.CACHE_STATUS_QPS_LATENCY_METRIC,
                time.time() * 1000 - start_time,
            )
            return result

        except Exception as e:
            logging.error(f"Get cache status failed: {e}")
            return {"error": f"Failed to get cache status: {str(e)}"}

    async def get_worker_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get worker status from gRPC server"""
        try:
            start_time = time.time() * 1000
            await self._ensure_connection()
            request = pb2.StatusVersionPB(
                latest_cache_version=query_params.get("latest_cache_version", -1),
                latest_finished_version=query_params.get("latest_finished_version", -1),
            )
            response = await self.stub.GetWorkerStatus(request, timeout=1)
            # Convert response to dict format expected by frontend
            result = MessageToDict(
                response,
                preserving_proto_field_name=True,
                including_default_value_fields=True,
            )
            kmonitor.report(AccMetrics.WORKER_STATUS_QPS_METRIC, 1)
            kmonitor.report(
                GaugeMetrics.WORKER_STATUS_QPS_LANTENCY_METRIC,
                time.time() * 1000 - start_time,
            )
            return result
        except Exception as e:
            logging.error(f"Get worker status failed: {e}")
            return {"error": f"Failed to get worker status: {str(e)}"}

    async def set_log_level(self, req: Any) -> Dict[str, Any]:
        """Set log level - this would need to be implemented based on your requirements"""
        try:
            await self._ensure_connection()
            if isinstance(req, str):
                req = json.loads(req)
            request = pb2.SetLogLevelRequestPB(
                log_level=req.get("log_level", "INFO"),
            )
            await self.stub.SetLogLevel(request, timeout=3)
            return {"status": "ok"}
        except Exception as e:
            logging.error(f"Set log level failed: {e}")
            return {"error": f"Failed to set log level: {str(e)}"}

    async def update_eplb_config(self, req: Any) -> Dict[str, Any]:
        """Update EPLB config - this would need to be implemented based on your requirements"""
        try:
            await self._ensure_connection()
            if isinstance(req, str):
                req = json.loads(req)
            epld_req = pb2.UpdateEplbConfigRequestPB(
                mode=req.get("mode", "NONE"),
                update_time=int(time.time()),
            )
            await self.stub.UpdateEplbConfig(epld_req)
            return {"status": "ok"}
        except Exception as e:
            logging.error(f"Update EPLB config failed: {e}")
            return {"error": f"Failed to update EPLB config: {str(e)}"}

    async def update_scheduler_info(self, req: Any) -> Dict[str, Any]:
        """Update scheduler info - this would need to be implemented based on your requirements"""
        try:
            await self._ensure_connection()
            if isinstance(req, str):
                req = json.loads(req)
            update_schedule_info_req = pb2.UpdateSchedulerInfoRequestPB(
                scheduler_info=json.dumps(req)
            )
            await self.stub.UpdateSchedulerInfo(update_schedule_info_req)

            return {"status": "ok"}
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
            elif uri == "set_log_level":
                return await self.set_log_level(req)
            elif uri == "update_eplb_config":
                return await self.update_eplb_config(req)
            elif uri == "update_scheduler_info":
                return await self.update_scheduler_info(req)
            else:
                # Default case - return empty success
                return {"status": "ok"}
        except Exception as e:
            logging.error(f"POST request to {uri} failed: {e}")
            return {"error": f"Request failed: {str(e)}"}
