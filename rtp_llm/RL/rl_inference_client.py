import asyncio
import functools
import logging
from typing import Any, Callable, Dict, List, Optional

import grpc
import grpc.aio
from grpc import StatusCode

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CheckHealthResponsePB,
    EmptyPB,
    UpdateWeightsRequestPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub


class AsyncRpcInferenceClient:
    # Common gRPC channel options
    _GRPC_OPTIONS = [
        ("grpc.max_metadata_size", 1024 * 1024 * 1024),
        ("grpc.max_send_message_length", 1024 * 1024 * 1024),
        ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
    ]

    def __init__(self, members: List[str]):
        self._server_addresses = members
        logging.info(
            f"Initialized async RPC client with servers: {self._server_addresses}"
        )

    def _create_grpc_channel(self, address: str) -> grpc.aio.Channel:
        """Create an async gRPC channel with common options"""
        return grpc.aio.insecure_channel(address, options=self._GRPC_OPTIONS)

    async def _send_request(
        self,
        address: str,
        rpc_func: Callable[[RpcServiceStub, Any, Optional[float]], Any],
        request: Any,
        timeout: Optional[float] = None,
        operation_name: str = "request",
    ) -> Any:
        """
        Generic method to send a single async request to a server

        :param address: Target server address
        :param rpc_func: Async RPC function to call
        :param request: Request object
        :param timeout: Request timeout
        :param operation_name: Operation name for logging
        :return: Response from server
        """
        channel = None
        try:
            channel = self._create_grpc_channel(address)
            stub = RpcServiceStub(channel)

            if timeout:
                response = await rpc_func(stub, request, timeout)
            else:
                response = await rpc_func(stub, request, None)

            logging.debug(f"Successfully sent {operation_name} to {address}")
            return response

        except grpc.aio.AioRpcError as e:
            status_code = e.code()
            details = f"gRPC error to {address}: [{status_code.name}] {e.details()}"
            if status_code == StatusCode.DEADLINE_EXCEEDED:
                raise TimeoutError(
                    f"{operation_name.capitalize()} to {address} timed out"
                ) from e
            raise ConnectionError(details) from e
        finally:
            if channel:
                await channel.close()

    async def _broadcast_request(
        self,
        rpc_func: Callable[[RpcServiceStub, Any, Optional[float]], Any],
        request: Any,
        timeout: Optional[float] = None,
        operation_name: str = "request",
        return_response_value: bool = False,
        response_extractor: Optional[Callable[[Any], Any]] = None,
        error_default: Any = None,
    ) -> Dict[str, Any]:
        """
        Generic method to broadcast an async request to all servers concurrently

        :param rpc_func: Async RPC function to call
        :param request: Request object
        :param timeout: Request timeout
        :param operation_name: Operation name for logging
        :param return_response_value: Whether to return response values instead of bool
        :param response_extractor: Function to extract value from response
        :param error_default: Default value to return on error
        :return: Dictionary mapping server address to result
        """
        # Create async tasks for all servers
        tasks = []
        address_mapping = {}
        for address in self._server_addresses:
            task = asyncio.create_task(
                self._send_request(address, rpc_func, request, timeout, operation_name)
            )
            tasks.append(task)
            address_mapping[task] = address

        # Wait for all tasks to complete using asyncio.gather with return_exceptions=True
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for task, result in zip(tasks, task_results):
            addr = address_mapping[task]

            if isinstance(result, Exception):
                logging.error(
                    f"{operation_name.capitalize()} to {addr} failed: {str(result)}"
                )
                if return_response_value:
                    results[addr] = error_default
                else:
                    results[addr] = False
            else:
                if return_response_value and response_extractor:
                    results[addr] = response_extractor(result)
                elif return_response_value:
                    results[addr] = result
                else:
                    results[addr] = True

        return results

    async def update_weights(
        self,
        model_name: str,
        description: str,
        method_name: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, bool]:
        """
        Broadcast weight update request to all servers concurrently

        :param model_name: Model name for the weight update
        :param description: Description of the weight update
        :param method_name: Optimization method name
        :param timeout: Request timeout in seconds
        :return: Dictionary mapping server address to success status
        """
        request = UpdateWeightsRequestPB(
            name=model_name, desc=description, method=method_name
        )

        async def rpc_func(
            stub: RpcServiceStub,
            req: UpdateWeightsRequestPB,
            timeout_val: Optional[float],
        ):
            if timeout_val:
                return await stub.UpdateWeights(req, timeout=timeout_val)
            else:
                return await stub.UpdateWeights(req)

        return await self._broadcast_request(
            rpc_func, request, timeout, "weight update request"
        )

    async def pause_servers(
        self,
        address: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, bool]:
        """
        Send pause request to specified server or broadcast to all servers concurrently

        :param address: Target server address, if None will broadcast to all servers
        :param timeout: Request timeout in seconds
        :return: Dictionary mapping server address to success status
        """
        request = EmptyPB()

        async def rpc_func(
            stub: RpcServiceStub, req: EmptyPB, timeout_val: Optional[float]
        ):
            if timeout_val:
                return await stub.SetPause(req, timeout=timeout_val)
            else:
                return await stub.SetPause(req)

        if address:
            # Send to specific server
            try:
                await self._send_request(
                    address, rpc_func, request, timeout, "pause request"
                )
                return {address: True}
            except Exception as e:
                logging.error(f"Pause request to {address} failed: {str(e)}")
                return {address: False}
        else:
            # Broadcast to all servers
            return await self._broadcast_request(
                rpc_func, request, timeout, "pause request"
            )

    async def restart_servers(
        self,
        address: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, bool]:
        """
        Send restart request to specified server or broadcast to all servers concurrently

        :param address: Target server address, if None will broadcast to all servers
        :param timeout: Request timeout in seconds
        :return: Dictionary mapping server address to success status
        """
        request = EmptyPB()

        async def rpc_func(
            stub: RpcServiceStub, req: EmptyPB, timeout_val: Optional[float]
        ):
            if timeout_val:
                return await stub.SetRestart(req, timeout=timeout_val)
            else:
                return await stub.SetRestart(req)

        if address:
            # Send to specific server
            try:
                await self._send_request(
                    address, rpc_func, request, timeout, "restart request"
                )
                return {address: True}
            except Exception as e:
                logging.error(f"Restart request to {address} failed: {str(e)}")
                return {address: False}
        else:
            # Broadcast to all servers
            return await self._broadcast_request(
                rpc_func, request, timeout, "restart request"
            )

    async def check_health(
        self,
        address: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, str]:
        """
        Check health status of specified server or all servers concurrently

        :param address: Target server address, if None will check all servers
        :param timeout: Request timeout in seconds
        :return: Dictionary mapping server address to health status
        """
        request = EmptyPB()

        async def rpc_func(
            stub: RpcServiceStub, req: EmptyPB, timeout_val: Optional[float]
        ):
            if timeout_val:
                return await stub.CheckHealth(req, timeout=timeout_val)
            else:
                return await stub.CheckHealth(req)

        if address:
            # Check specific server
            try:
                response = await self._send_request(
                    address, rpc_func, request, timeout, "health check"
                )
                return {address: response.health}
            except Exception as e:
                logging.error(f"Health check request to {address} failed: {str(e)}")
                return {address: "UNHEALTHY"}
        else:
            # Check all servers concurrently
            return await self._broadcast_request(
                rpc_func,
                request,
                timeout,
                "health check",
                return_response_value=True,
                response_extractor=lambda resp: resp.health,
                error_default="UNHEALTHY",
            )

    async def close(self):
        """Clean up resources"""
        # No explicit cleanup needed for async channels as they're closed in _send_request
        pass


# 使用示例
async def main():
    # 初始化异步客户端
    client = AsyncRpcInferenceClient(["127.0.0.1:26001", "127.0.0.1:26002"])

    try:
        # 1. 并发检查所有服务器健康状态
        print("Checking server health concurrently...")
        health_results = await client.check_health(timeout=10.0)
        for addr, status in health_results.items():
            print(f"Server {addr} health: {status}")

        # 2. 并发暂停所有服务器
        print("Pausing all servers concurrently...")
        pause_results = await client.pause_servers(timeout=10.0)
        if all(pause_results.values()):
            print("All servers paused successfully")
        else:
            failed_servers = [k for k, v in pause_results.items() if not v]
            print(f"Failed to pause servers: {failed_servers}")

        # 3. 并发重启所有服务器
        print("Restarting all servers concurrently...")
        restart_results = await client.restart_servers(timeout=10.0)
        if all(restart_results.values()):
            print("All servers restarted successfully")
        else:
            failed_servers = [k for k, v in restart_results.items() if not v]
            print(f"Failed to restart servers: {failed_servers}")

        # 4. 并发广播权重更新请求
        print("Broadcasting weight update concurrently...")
        update_results = await client.update_weights(
            model_name="example_model",
            description="weight_update_001",
            method_name="adam_optimizer",
            timeout=30.0,
        )

        # 检查所有节点是否成功
        if all(update_results.values()):
            print("All servers updated successfully")
        else:
            failed_servers = [k for k, v in update_results.items() if not v]
            print(f"Failed to update on servers: {failed_servers}")

        # 5. 单独操作特定服务器
        print("Checking specific server health...")
        specific_health = await client.check_health(
            address="127.0.0.1:26001", timeout=5.0
        )
        print(f"Specific server health: {specific_health}")

        # 6. 演示高并发能力 - 同时执行多种操作
        print("Demonstrating high concurrency...")
        health_task = client.check_health(timeout=5.0)
        pause_task = client.pause_servers(address="127.0.0.1:26001", timeout=5.0)
        restart_task = client.restart_servers(address="127.0.0.1:26002", timeout=5.0)

        # 并发执行多个不同操作
        concurrent_results = await asyncio.gather(
            health_task, pause_task, restart_task, return_exceptions=True
        )

        print("Concurrent operation results:")
        for i, result in enumerate(concurrent_results):
            if isinstance(result, Exception):
                print(f"Operation {i+1} failed: {result}")
            else:
                print(f"Operation {i+1} succeeded: {result}")

    finally:
        await client.close()


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
