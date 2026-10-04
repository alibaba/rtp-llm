import asyncio
import json
import logging
import time
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

import grpc
from google.protobuf.json_format import MessageToDict

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.frontend.sleep_validation import dedupe_addresses
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.utils.lifecycle_controller import LifecycleController
from rtp_llm.utils.lifecycle_rpc import ControlRpcTransport
from rtp_llm.utils.time_util import Timer


class GrpcClientWrapper:
    """Serving/maintenance RPCs and a thin adapter to lifecycle control."""

    def __init__(
        self,
        server_port: int,
        dp_addresses: Optional[List[str]] = None,
        client_config: Optional[Dict[str, int]] = None,
        control_addresses: Optional[List[str]] = None,
        expected_control_address_count: Optional[int] = None,
        control_address_resolver: Optional[Callable[[], List[str]]] = None,
        lifecycle_store: Optional[Any] = None,
        lifecycle_store_factory: Optional[Callable[[], Optional[Any]]] = None,
        require_instance_lease: bool = False,
    ):
        self.server_port = server_port
        self.address = f"localhost:{server_port}"
        self.channel = None
        self.stub = None
        # Serving-route broadcast targets, normally one representative per DP
        # group. Do not use these for sleep/wake_up: lifecycle control must
        # reach every backend rank process that owns GPU resources.
        self.dp_addresses = dedupe_addresses(
            dp_addresses if dp_addresses else [self.address]
        )
        self._client_config = client_config or {}
        # One pool for addressed control/maintenance RPCs, separate from health.
        self._control_rpc = ControlRpcTransport(self._client_config)
        self._lifecycle = LifecycleController(
            self._control_rpc,
            control_addresses or [self.address],
            expected_control_address_count=expected_control_address_count,
            control_address_resolver=control_address_resolver,
            lifecycle_store=lifecycle_store,
            lifecycle_store_factory=lifecycle_store_factory,
            require_instance_lease=require_instance_lease,
        )

    async def _ensure_connection(self):
        """Ensure gRPC channel and stub are created"""
        if self.channel is None or self.stub is None:
            self.channel = grpc.aio.insecure_channel(
                self.address,
                options=[(k, v) for k, v in self._client_config.items()],
            )
            self.stub = RpcServiceStub(self.channel)

    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None
        await self._control_rpc.close()

    async def _reset_main_channel(self) -> None:
        """Tear down ONLY the health/status channel so the next probe reconnects.

        Deliberately does not touch self._control_rpc.channels: those carry in-flight
        sleep/wake lifecycle RPCs. Closing a channel while one of its calls is
        genuinely in-flight (server-accepted, still processing) raises
        asyncio.CancelledError into the awaiting coroutine -- a BaseException
        that bypasses every ``except Exception`` on the lifecycle path. A
        routine health-probe timeout during a sleep/wake drain would otherwise
        cancel the unrelated lifecycle operation and surface HTTP 500 while the
        backend keeps transitioning to SLEEPING -- a control-plane split brain.
        """
        if self.channel:
            try:
                await self.channel.close()
            except Exception as e:
                logging.warning(f"Failed to close health channel: {e}")
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
            # Reset only the health channel. Never close the shared lifecycle
            # (the control transport) here: see _reset_main_channel -- doing so would
            # cancel an in-flight sleep/wake RPC that shares this wrapper.
            await self._reset_main_channel()
            return {
                "status": "error",
                "message": e,
            }

    async def get_cache_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get cache status from gRPC server"""
        try:
            start_time = perf_counter()
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
                (perf_counter() - start_time) * 1000.0,
            )
            return result

        except Exception as e:
            logging.error(f"Get cache status failed: {e}")
            return {"error": f"Failed to get cache status: {str(e)}"}

    async def get_worker_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get worker status from gRPC server"""
        try:
            start_time = perf_counter()
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
                (perf_counter() - start_time) * 1000.0,
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

    async def sleep_serving(self, req: Any) -> Dict[str, Any]:
        return await self._lifecycle.sleep_serving(req)

    async def wake_up_serving(self, req: Any = None) -> Dict[str, Any]:
        return await self._lifecycle.wake_up_serving(req)

    async def get_sleep_status(self, req: Any = None) -> Dict[str, Any]:
        return await self._lifecycle.get_sleep_status(req)

    async def is_sleeping(self, req: Any = None) -> Dict[str, Any]:
        return await self._lifecycle.is_sleeping(req)

    async def start_profile(self, req: Any) -> Dict[str, Any]:
        """Start profiling switch in backend process"""
        try:
            await self._ensure_connection()
            if isinstance(req, str):
                req = json.loads(req)
            if req is None:
                req = {}
            request = pb2.StartProfileRequestPB(
                trace_name=str(req.get("trace_name", "")),
                start_step=int(req.get("start_step", 0)),
                num_steps=int(req.get("num_steps", 0)),
                enable_all_rank=bool(
                    req.get("enable_all_rank", req.get("all_tp", False))
                ),
            )
            await self.stub.StartProfile(request, timeout=3)
            return {"status": "ok"}

        except Exception as e:
            logging.error(f"Start profile failed: {e}")
            return {"error": f"Failed to start profile: {str(e)}"}

    async def dump_torch_allocator(self) -> Dict[str, Any]:
        """Trigger one allocator dump in every backend process across DP/TP."""

        async def send_to_address(address: str):
            await self._control_rpc.ensure(address)
            return await self._control_rpc.stubs[address].DumpTorchAllocator(
                pb2.EmptyPB(), timeout=90
            )

        addresses = self.dp_addresses
        responses = await asyncio.gather(
            *(send_to_address(address) for address in addresses),
            return_exceptions=True,
        )

        backends = []
        errors = []
        for address, response in zip(addresses, responses):
            if isinstance(response, Exception):
                errors.append(f"{address}: {response}")
                continue
            if not response.results:
                errors.append(f"{address}: backend returned no allocator dump results")
                continue
            for result in response.results:
                result_dict = {
                    "world_rank": result.world_rank,
                    "dp_rank": result.dp_rank,
                    "tp_rank": result.tp_rank,
                    "local_rank": result.local_rank,
                    "pid": result.pid,
                    "success": result.success,
                    "file_path": result.file_path,
                    "error": result.error,
                    "dp_address": address,
                }
                backends.append(result_dict)
                if not result.success:
                    errors.append(
                        f"{address}/world_rank={result.world_rank}: {result.error}"
                    )

        return {
            "status": "ok" if not errors else "error",
            "backends": backends,
            "errors": errors,
        }

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
        """Update scheduler info on all DP addresses"""
        try:
            if isinstance(req, str):
                req = json.loads(req)
            update_schedule_info_req = pb2.UpdateSchedulerInfoRequestPB(
                scheduler_info=json.dumps(req)
            )

            async def send_to_address(address: str):
                await self._control_rpc.ensure(address)
                await self._control_rpc.stubs[address].UpdateSchedulerInfo(
                    update_schedule_info_req
                )

            tasks = [send_to_address(addr) for addr in self.dp_addresses]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            errors = [
                f"{self.dp_addresses[i]}: {str(r)}"
                for i, r in enumerate(results)
                if isinstance(r, Exception)
            ]
            if errors:
                logging.error(
                    f"Update scheduler info failed on some addresses: {errors}"
                )
                return {"error": f"Failed on some addresses: {errors}"}

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
            elif uri == "sleep":
                return await self.sleep_serving(req)
            elif uri == "wake_up":
                return await self.wake_up_serving(req)
            elif uri == "is_sleeping":
                return await self.is_sleeping(req)
            elif uri == "sleep_status":
                return await self.get_sleep_status(req)
            elif uri == "start_profile":
                return await self.start_profile(req)
            elif uri == "dump_torch_allocator":
                return await self.dump_torch_allocator()
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
