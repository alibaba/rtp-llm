"""Transport for lifecycle control RPCs."""

import asyncio
import logging
from time import perf_counter
from typing import Any, Dict, List

import grpc
from google.protobuf.json_format import MessageToDict

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.utils.sleep_timing import log_sleep_timing


class LifecycleRpcTransport:
    """Own lifecycle channels and fan-out; policy remains in the controller."""

    def __init__(self, client_config: Dict[str, int]):
        self._client_config = client_config
        self.channels: Dict[str, Any] = {}
        self.stubs: Dict[str, Any] = {}

    async def ensure(self, address: str) -> None:
        if address not in self.channels or self.stubs.get(address) is None:
            self.channels[address] = grpc.aio.insecure_channel(
                address,
                options=list(self._client_config.items()),
            )
            self.stubs[address] = RpcServiceStub(self.channels[address])

    async def call(
        self, address: str, rpc_name: str, request: Any, timeout_s: float
    ) -> Dict[str, Any]:
        started = perf_counter()
        operation = {"SleepServing": "sleep", "WakeUpServing": "wake"}.get(
            rpc_name, "control"
        )
        try:
            await self.ensure(address)
            response = await getattr(self.stubs[address], rpc_name)(
                request, timeout=timeout_s
            )
            result: Dict[str, Any] = {"address": address, "status": "ok"}
            if response is not None and not isinstance(response, pb2.EmptyPB):
                result.update(
                    MessageToDict(
                        response,
                        preserving_proto_field_name=True,
                        including_default_value_fields=True,
                    )
                )
            log_sleep_timing(
                operation,
                rpc_name,
                (perf_counter() - started) * 1000.0,
                scope="controller_rpc",
                fields={"address": address, "rpc": rpc_name},
            )
            return result
        except grpc.aio.AioRpcError as e:
            grpc_status = e.code().name
            error = str(e.details())
            logging.error("%s failed on %s: %s", rpc_name, address, error)
            log_sleep_timing(
                operation,
                rpc_name,
                (perf_counter() - started) * 1000.0,
                status="error",
                scope="controller_rpc",
                fields={
                    "address": address,
                    **(
                        {"grpc_status": grpc_status}
                        if grpc_status
                        else {"error_type": type(e).__name__}
                    ),
                },
            )
            return {
                "address": address,
                "error": error,
                "grpc_status": grpc_status,
            }
        except Exception as e:
            logging.error("%s failed on %s: %s", rpc_name, address, e)
            log_sleep_timing(
                operation,
                rpc_name,
                (perf_counter() - started) * 1000.0,
                status="error",
                scope="controller_rpc",
                fields={"address": address, "error_type": type(e).__name__},
            )
            return {"address": address, "error": str(e)}

    async def broadcast(
        self,
        addresses: List[str],
        rpc_name: str,
        request: Any,
        timeout_s: float,
    ) -> List[Dict[str, Any]]:
        started = perf_counter()
        results = await asyncio.gather(
            *(self.call(address, rpc_name, request, timeout_s) for address in addresses)
        )
        operation = {"SleepServing": "sleep", "WakeUpServing": "wake"}.get(
            rpc_name, "control"
        )
        log_sleep_timing(
            operation,
            f"broadcast_{rpc_name}",
            (perf_counter() - started) * 1000.0,
            scope="controller",
            fields={
                "address_count": len(addresses),
                "error_count": sum("error" in result for result in results),
            },
        )
        return results

    async def close(self) -> None:
        for address, channel in self.channels.items():
            try:
                await channel.close()
            except Exception as e:
                logging.warning(
                    "Failed to close lifecycle channel for %s: %s", address, e
                )
        self.channels.clear()
        self.stubs.clear()
