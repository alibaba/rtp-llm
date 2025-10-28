from urllib import response

import grpc
import torch

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc


def print_health(response):
    print("\n=== response Fields ===")
    # print(f"Response received: {response}")


def test_grpc():
    # 1. 连接到 gRPC 服务器
    channel = grpc.insecure_channel("localhost:27001")  # 替换为你的服务端口
    stub = pb2_grpc.EmbeddingRpcServiceStub(channel)

    # 2. 构造请求
    import time

    # print(int(time.time()))
    request = pb2.EmbeddingHealthRequestPB(
        request_id=0,
    )

    try:
        # 3. 调用接口
        response = stub.health(request)
        print_health(response)

    except grpc.RpcError as e:
        print(f"RPC failed: {e.code()}: {e.details()}")


from typing import Optional

import numpy as np
import torch


def tensor_pb_to_torch(tensor_pb) -> Optional[torch.Tensor]:
    """
    将 TensorPB 转换为 PyTorch Tensor
    返回: torch.Tensor 或 None（转换失败时）
    """
    # 获取数据类型和形状
    data_type = tensor_pb.data_type
    shape = list(tensor_pb.shape)  # 转换为 list[int]

    # 根据数据类型选择对应的字节字段
    if data_type == tensor_pb.DataType.FP32:
        dtype_np = np.float32
        buffer = tensor_pb.fp32_data
    elif data_type == tensor_pb.DataType.INT32:
        dtype_np = np.int32
        buffer = tensor_pb.int32_data
    elif data_type == tensor_pb.DataType.FP16:
        # FP16 需要特殊处理（numpy 没有 float16 的 frombuffer 直接支持）
        buffer = tensor_pb.fp16_data
        np_array = np.frombuffer(buffer, dtype=np.uint16).view(np.float16)
        torch_tensor = torch.from_numpy(np_array).reshape(shape)
        return torch_tensor
    elif data_type == tensor_pb.DataType.BF16:
        # BF16 需要转换为 float32 再处理
        buffer = tensor_pb.bf16_data
        np_array = np.frombuffer(buffer, dtype=np.uint16)
        np_array = np_array.astype(np.float32) / 32768.0  # 简化的 BF16 转换
        torch_tensor = torch.from_numpy(np_array).reshape(shape)
        return torch_tensor
    else:
        print(f"Unsupported data type: {data_type}")
        return None

    # 检查数据是否为空
    if not buffer:
        print("Empty data buffer")
        return None

    # 将字节数据转换为 numpy 数组
    try:
        np_array = np.frombuffer(buffer, dtype=dtype_np)
    except Exception as e:
        print(f"Failed to parse buffer: {e}")
        return None

    # 检查形状是否匹配
    if np.prod(shape) != np_array.size:
        print(f"Shape {shape} does not match data size {np_array.size}")
        return None

    # 转换为 PyTorch Tensor 并调整形状
    torch_tensor = torch.from_numpy(np_array).reshape(shape)
    return torch_tensor


def test_grpc_input():
    # 1. 连接到 gRPC 服务器

    channel = grpc.insecure_channel("localhost:27001")
    stub = pb2_grpc.EmbeddingRpcServiceStub(channel)

    # 2. 构造Tensor数据 -------------------------------------------------
    # 创建示例torch tensor
    img_tensor = torch.randn(3, 224, 224)  # 假设是图像特征

    def torch_to_tensorpb(tensor: torch.Tensor) -> pb2.TensorPB:
        """将torch tensor转换为符合proto定义的TensorPB"""
        np_array = tensor.cpu().numpy()

        # 创建基础TensorPB对象
        tensor_pb = pb2.TensorPB(
            data_type=pb2.TensorPB.DataType.FP32,  # 初始值会被覆盖
            shape=list(np_array.shape),  # shape自动转换为int64
        )

        # 根据数据类型填充对应字段
        if tensor.dtype == torch.float32:
            tensor_pb.data_type = pb2.TensorPB.DataType.FP32
            tensor_pb.fp32_data = np_array.tobytes()
        elif tensor.dtype == torch.int32:
            tensor_pb.data_type = pb2.TensorPB.DataType.INT32
            tensor_pb.int32_data = np_array.tobytes()
        elif tensor.dtype == torch.float16:
            tensor_pb.data_type = pb2.TensorPB.DataType.FP16
            tensor_pb.fp16_data = np_array.tobytes()
        elif tensor.dtype == torch.bfloat16:
            tensor_pb.data_type = pb2.TensorPB.DataType.BF16
            tensor_pb.bf16_data = np_array.tobytes()
        else:
            raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")

        return tensor_pb

    # 3. 构造MultiModalFeatures -----------------------------------------
    multimodal_features = [
        pb2.MultimodalInputPB(
            multimodal_type=1,
            multimodal_tensor=torch_to_tensorpb(img_tensor),
            multimodal_url="https://img.ixintu.com/download/jpg/201911/e25b904bc42a74d7d77aed81e66d772c.jpg",
        )
    ]

    # 4. 构造EmbeddingInputPB请求 ---------------------------------------
    request = pb2.EmbeddingInputPB(
        token_ids=[
            151644,
            8948,
            198,
            2610,
            525,
            264,
            10950,
            17847,
            13,
            151645,
            198,
            151644,
            872,
            198,
            16,
            15,
            18947,
            18600,
            53481,
            45930,
            24669,
            220,
            16,
            25,
            220,
            151652,
            151655,
            151653,
            151645,
            198,
            151644,
            77091,
            198,
        ],  # 示例token ids
        token_type_ids=[
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 示例segment ids
        input_lengths=[33],  # 输入长度
        request_id=1,  # 唯一请求ID
        multimodal_features=multimodal_features,
    )

    try:
        # 调用 decode 方法（返回一个响应流）
        response_stream = stub.decode(request)  # 注意这里是流式响应

        # 遍历流式响应
        for response in response_stream:

            tensor_pb = response.output_t
            torch_tensor = tensor_pb_to_torch(tensor_pb)
            if torch_tensor is not None:
                print("Tensor shape:", torch_tensor.shape)
                print("Tensor dtype:", torch_tensor.dtype)
                print("Tensor data (first 5 elements):", torch_tensor.flatten()[:5])

    except grpc.RpcError as e:
        print(f"RPC failed: {e.code()}: {e.details()}")


# 辅助函数：打印响应（根据实际响应结构修改）
def print_health(response):
    print(f"Health Status: {response}")


if __name__ == "__main__":
    test_grpc()
    test_grpc_input()
