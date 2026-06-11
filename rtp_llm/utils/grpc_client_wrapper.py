import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import grpc
from google.protobuf.json_format import MessageToDict

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.utils.time_util import Timer


def _dedupe_addresses(addresses: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for address in addresses:
        if address in seen:
            continue
        seen.add(address)
        deduped.append(address)
    return deduped


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class GrpcClientWrapper:
    """Wrapper for direct gRPC calls to replace async_request_server"""

    def __init__(
        self,
        server_port: int,
        dp_addresses: Optional[List[str]] = None,
        control_addresses: Optional[List[str]] = None,
    ):
        self.server_port = server_port
        self.address = f"localhost:{server_port}"
        self.channel = None
        self.stub = None
        # Serving-route broadcast targets, normally one representative per DP
        # group. Do not use these for freeze/resume: lifecycle control must
        # reach every backend rank process that owns GPU resources.
        self.dp_addresses = _dedupe_addresses(
            dp_addresses if dp_addresses else [self.address]
        )
        self.control_addresses = _dedupe_addresses(
            control_addresses if control_addresses else [self.address]
        )
        self._lifecycle_lock = asyncio.Lock()
        self._dp_channels: Dict[str, Any] = {}
        self._dp_stubs: Dict[str, Any] = {}

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

    async def _ensure_dp_connection(self, address: str):
        """Ensure gRPC channel and stub are created for a specific DP address"""
        if address not in self._dp_channels or self._dp_stubs.get(address) is None:
            self._dp_channels[address] = grpc.aio.insecure_channel(
                address,
                options=[
                    ("grpc.max_metadata_size", 1024 * 1024 * 1024),
                ],
            )
            self._dp_stubs[address] = RpcServiceStub(self._dp_channels[address])

    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None
        for address, channel in self._dp_channels.items():
            try:
                await channel.close()
            except Exception as e:
                logging.warning(f"Failed to close DP channel for {address}: {e}")
        self._dp_channels.clear()
        self._dp_stubs.clear()

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
        """Get cache status from all DP backends.

        Returns a list of per-DP results under the ``"results"`` key.
        If no backend is reachable, returns ``{"error": ...}``.
        """
        start_time = time.time() * 1000
        request = pb2.CacheVersionPB(
            latest_cache_version=query_params.get("latest_cache_version", -1),
            need_cache_keys=query_params.get("need_cache_keys", True),
        )
        results = []
        for addr in self.dp_addresses:
            try:
                await self._ensure_dp_connection(addr)
                response = await self._dp_stubs[addr].GetCacheStatus(request, timeout=1)
                result = MessageToDict(
                    response,
                    preserving_proto_field_name=True,
                    including_default_value_fields=True,
                )
                result["address"] = addr
                results.append(result)
            except Exception as e:
                logging.warning(f"Get cache status from {addr} failed: {e}")
        kmonitor.report(AccMetrics.CACHE_STATUS_QPS_METRIC, 1)
        kmonitor.report(
            GaugeMetrics.CACHE_STATUS_QPS_LATENCY_METRIC,
            time.time() * 1000 - start_time,
        )
        if not results:
            return {"error": "No backend available for cache_status"}
        # For backward compatibility, merge first result's fields at top level
        merged = dict(results[0])
        merged["results"] = results
        merged["dp_size"] = len(results)
        return merged

    async def get_worker_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get worker status from all DP backends.

        Returns a list of per-DP results under the ``"results"`` key.
        If no backend is reachable, returns ``{"error": ...}``.
        """
        start_time = time.time() * 1000
        request = pb2.StatusVersionPB(
            latest_cache_version=query_params.get("latest_cache_version", -1),
            latest_finished_version=query_params.get("latest_finished_version", -1),
        )
        results = []
        for addr in self.dp_addresses:
            try:
                await self._ensure_dp_connection(addr)
                response = await self._dp_stubs[addr].GetWorkerStatus(
                    request, timeout=1
                )
                result = MessageToDict(
                    response,
                    preserving_proto_field_name=True,
                    including_default_value_fields=True,
                )
                result["address"] = addr
                results.append(result)
            except Exception as e:
                logging.warning(f"Get worker status from {addr} failed: {e}")
        kmonitor.report(AccMetrics.WORKER_STATUS_QPS_METRIC, 1)
        kmonitor.report(
            GaugeMetrics.WORKER_STATUS_QPS_LANTENCY_METRIC,
            time.time() * 1000 - start_time,
        )
        if not results:
            return {"error": "No backend available for worker_status"}
        merged = dict(results[0])
        merged["results"] = results
        merged["dp_size"] = len(results)
        return merged

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

    async def _call_control_rpc(
        self, address: str, rpc_name: str, request: Any, timeout_s: float
    ) -> Dict[str, Any]:
        try:
            await self._ensure_dp_connection(address)
            rpc = getattr(self._dp_stubs[address], rpc_name)
            response = await rpc(request, timeout=timeout_s)
            result: Dict[str, Any] = {"address": address, "status": "ok"}
            if response is not None and not isinstance(response, pb2.EmptyPB):
                result.update(
                    MessageToDict(
                        response,
                        preserving_proto_field_name=True,
                        including_default_value_fields=True,
                    )
                )
            return result
        except grpc.aio.AioRpcError as e:
            logging.error("%s failed on %s: %s", rpc_name, address, e.details())
            return {
                "address": address,
                "error": str(e.details()),
                "grpc_status": e.code().name,
            }
        except Exception as e:
            logging.error("%s failed on %s: %s", rpc_name, address, e)
            return {"address": address, "error": str(e)}

    async def _broadcast_control_rpc(
        self, rpc_name: str, request: Any, timeout_s: float
    ) -> List[Dict[str, Any]]:
        tasks = [
            self._call_control_rpc(address, rpc_name, request, timeout_s)
            for address in self.control_addresses
        ]
        return await asyncio.gather(*tasks)

    def _aggregate_freeze_status(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successes = [result for result in results if "error" not in result]
        failures = [result for result in results if "error" in result]
        if not successes:
            return {
                "error": "Failed to get freeze status from all control ranks",
                "results": results,
                "rank_count": len(results),
                "rank_success_count": 0,
                "state": "MIXED",
            }

        states = {str(result.get("state", "")) for result in successes}
        gpu_states = {str(result.get("gpu_resource_state", "")) for result in successes}
        kv_states = {str(result.get("kv_memory_state", "")) for result in successes}
        aggregate = dict(successes[0])
        aggregate["state"] = (
            next(iter(states)) if len(states) == 1 and not failures else "MIXED"
        )
        aggregate["gpu_resource_state"] = (
            next(iter(gpu_states)) if len(gpu_states) == 1 and not failures else "MIXED"
        )
        aggregate["kv_memory_state"] = (
            next(iter(kv_states)) if len(kv_states) == 1 and not failures else "MIXED"
        )
        aggregate["freeze_epoch"] = max(
            _as_int(result.get("freeze_epoch", 0)) for result in successes
        )
        aggregate["active_request_count"] = sum(
            _as_int(result.get("active_request_count", 0)) for result in successes
        )
        aggregate["active_cache_transfer_count"] = sum(
            _as_int(result.get("active_cache_transfer_count", 0))
            for result in successes
        )
        aggregate["device_kv_cache_valid"] = all(
            bool(result.get("device_kv_cache_valid", False)) for result in successes
        )
        aggregate["rank_count"] = len(results)
        aggregate["rank_success_count"] = len(successes)
        aggregate["results"] = results
        aggregate.pop("address", None)
        aggregate.pop("status", None)
        if failures:
            aggregate["error"] = "Failed to get freeze status from some control ranks"
            aggregate["grpc_status"] = failures[0].get("grpc_status", "UNAVAILABLE")
        return aggregate

    async def freeze_serving(self, req: Any) -> Dict[str, Any]:
        """Trigger engine freeze on every lifecycle control rank."""
        async with self._lifecycle_lock:
            return await self._freeze_serving_locked(req)

    async def _freeze_serving_locked(self, req: Any) -> Dict[str, Any]:
        try:
            if isinstance(req, str):
                req = json.loads(req)
            if req is None:
                req = {}
            mode = str(req.get("mode", "graceful"))
            drain_timeout_ms = int(req.get("drain_timeout_ms", 0))
            request = pb2.FreezeRequestPB(
                mode=mode,
                drain_timeout_ms=drain_timeout_ms,
                force=(mode == "force"),
                reason=str(req.get("reason", "")),
            )
            prepare_request = pb2.FreezeRequestPB()
            prepare_request.CopyFrom(request)
            prepare_request.prepare_only = True
            commit_request = pb2.FreezeRequestPB()
            commit_request.CopyFrom(request)
            commit_request.commit_only = True
            commit_request.drain_timeout_ms = 0

            # prepare blocks on drain; leave headroom on top of drain timeout.
            # Only after every rank is drained do we send commit, avoiding a
            # half-frozen instance when one rank times out.
            timeout_s = max(60.0, drain_timeout_ms / 1000.0 + 30.0)
            prepare_results = await self._broadcast_control_rpc(
                "FreezeServing", prepare_request, timeout_s
            )
            failures = [result for result in prepare_results if "error" in result]
            if failures:
                abort_results = await self._broadcast_control_rpc(
                    "ResumeServing", pb2.EmptyPB(), timeout_s=60
                )
                return {
                    "error": "Failed to prepare freeze on some control ranks",
                    "grpc_status": failures[0].get("grpc_status", "UNKNOWN"),
                    "results": prepare_results,
                    "abort_results": abort_results,
                    "rank_count": len(prepare_results),
                    "rank_success_count": len(prepare_results) - len(failures),
                }

            commit_results = await self._broadcast_control_rpc(
                "FreezeServing", commit_request, timeout_s=60
            )
            failures = [result for result in commit_results if "error" in result]
            if failures:
                status = await self.get_freeze_status()
                return {
                    "error": "Failed to commit freeze on some control ranks",
                    "grpc_status": failures[0].get("grpc_status", "UNKNOWN"),
                    "results": commit_results,
                    "freeze_status": status,
                    "rank_count": len(commit_results),
                    "rank_success_count": len(commit_results) - len(failures),
                }
            status = await self.get_freeze_status()
            if "error" in status:
                return status
            if status.get("state") != "FROZEN":
                return {
                    "error": "Freeze did not converge to FROZEN on all control ranks",
                    "grpc_status": "FAILED_PRECONDITION",
                    "freeze_status": status,
                    "rank_count": status.get("rank_count", len(commit_results)),
                    "rank_success_count": status.get(
                        "rank_success_count", len(commit_results)
                    ),
                }
            return {
                "status": "ok",
                "state": status.get("state", ""),
                "freeze_epoch": _as_int(status.get("freeze_epoch", 0)),
                "results": status.get("results", commit_results),
                "rank_count": status.get("rank_count", len(commit_results)),
                "rank_success_count": status.get(
                    "rank_success_count", len(commit_results)
                ),
            }
        except grpc.aio.AioRpcError as e:
            logging.error(f"Freeze serving failed: {e.details()}")
            return {
                "error": f"Failed to freeze serving: {e.details()}",
                "grpc_status": e.code().name,
            }
        except Exception as e:
            logging.error(f"Freeze serving failed: {e}")
            return {"error": f"Failed to freeze serving: {str(e)}"}

    async def resume_serving(self, req: Any = None) -> Dict[str, Any]:
        """Trigger engine resume on every lifecycle control rank."""
        async with self._lifecycle_lock:
            return await self._resume_serving_locked(req)

    async def _resume_serving_locked(self, req: Any = None) -> Dict[str, Any]:
        try:
            request = pb2.EmptyPB()
            results = await self._broadcast_control_rpc(
                "ResumeServing", request, timeout_s=600
            )
            failures = [result for result in results if "error" in result]
            if failures:
                return {
                    "error": "Failed to resume serving on some control ranks",
                    "grpc_status": failures[0].get("grpc_status", "UNKNOWN"),
                    "results": results,
                    "rank_count": len(results),
                    "rank_success_count": len(results) - len(failures),
                }
            status = await self.get_freeze_status()
            if "error" in status:
                return status
            if status.get("state") != "RUNNING":
                return {
                    "error": "Resume did not converge to RUNNING on all control ranks",
                    "grpc_status": "FAILED_PRECONDITION",
                    "freeze_status": status,
                    "rank_count": status.get("rank_count", len(results)),
                    "rank_success_count": status.get(
                        "rank_success_count", len(results)
                    ),
                }
            return {
                "status": "ok",
                "state": status.get("state", ""),
                "freeze_epoch": _as_int(status.get("freeze_epoch", 0)),
                "results": status.get("results", results),
                "rank_count": status.get("rank_count", len(results)),
                "rank_success_count": status.get("rank_success_count", len(results)),
            }
        except grpc.aio.AioRpcError as e:
            logging.error(f"Resume serving failed: {e.details()}")
            return {
                "error": f"Failed to resume serving: {e.details()}",
                "grpc_status": e.code().name,
            }
        except Exception as e:
            logging.error(f"Resume serving failed: {e}")
            return {"error": f"Failed to resume serving: {str(e)}"}

    async def get_freeze_status(self, req: Any = None) -> Dict[str, Any]:
        """Get aggregate freeze lifecycle status from every control rank."""
        try:
            request = pb2.EmptyPB()
            results = await self._broadcast_control_rpc(
                "GetFreezeStatus", request, timeout_s=3
            )
            return self._aggregate_freeze_status(results)
        except Exception as e:
            logging.error(f"Get freeze status failed: {e}")
            return {"error": f"Failed to get freeze status: {str(e)}"}

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
                await self._ensure_dp_connection(address)
                await self._dp_stubs[address].UpdateSchedulerInfo(
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
            elif uri == "freeze":
                return await self.freeze_serving(req)
            elif uri == "resume":
                return await self.resume_serving(req)
            elif uri == "freeze_status":
                return await self.get_freeze_status(req)
            elif uri == "start_profile":
                return await self.start_profile(req)
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
