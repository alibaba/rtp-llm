import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import TensorPB


def trans_option(pb_object, py_object, name):
    if getattr(py_object, name):
        getattr(pb_object, name).value = getattr(py_object, name)


def trans_option_cast(pb_object, py_object, name, func):
    if getattr(py_object, name):
        getattr(pb_object, name).value = func(getattr(py_object, name))


def trans_grpc_dtype(data_type: TensorPB.DataType):
    if data_type == TensorPB.DataType.FP32:
        return torch.float32
    elif data_type == TensorPB.DataType.INT32:
        return torch.int32
    elif data_type == TensorPB.DataType.FP16:
        return torch.float16
    elif data_type == TensorPB.DataType.BF16:
        return torch.bfloat16
    else:
        raise ValueError(f"unsupported TensorPB data_type: {data_type}")


def trans_tensor(t: TensorPB):
    shape = list(t.shape)
    dtype = trans_grpc_dtype(t.data_type)
    if t.data_type == TensorPB.DataType.FP32:
        payload = t.fp32_data
    elif t.data_type == TensorPB.DataType.INT32:
        payload = t.int32_data
    elif t.data_type == TensorPB.DataType.FP16:
        payload = t.fp16_data
    elif t.data_type == TensorPB.DataType.BF16:
        payload = t.bf16_data
    else:
        raise ValueError(f"unsupported TensorPB data_type: {t.data_type}")

    if not shape and not payload:
        return torch.empty((0,), dtype=dtype)

    numel = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError(
                f"TensorPB shape dimensions must be non-negative, got shape={shape}"
            )
        numel *= dimension

    if numel == 0:
        return torch.empty(shape, dtype=dtype)

    expected_bytes = numel * torch.empty((), dtype=dtype).element_size()
    actual_bytes = len(payload)
    if actual_bytes != expected_bytes:
        raise ValueError(
            "TensorPB payload byte length mismatch for "
            f"data_type={t.data_type}, shape={shape}: expected {expected_bytes} bytes, "
            f"got {actual_bytes}"
        )

    return torch.frombuffer(payload, dtype=dtype).reshape(shape)


def trans_from_tensor(t: torch.Tensor):
    if t is None or t.numel() == 0:
        return TensorPB()
    res = TensorPB()
    t = t.cpu()
    res.shape.extend(list(t.shape))
    if t.dtype == torch.float32:
        res.data_type = TensorPB.DataType.FP32
        res.fp32_data = t.numpy().tobytes()
    elif t.dtype == torch.int32:
        res.data_type = TensorPB.DataType.INT32
        res.int32_data = t.numpy().tobytes()
    elif t.dtype == torch.float16:
        res.data_type = TensorPB.DataType.FP16
        res.fp16_data = t.numpy().tobytes()
    elif t.dtype == torch.bfloat16:
        res.data_type = TensorPB.DataType.BF16
        res.bf16_data = t.view(torch.int16).numpy().tobytes()
    else:
        raise Exception("unknown tensor data type")
    return res
