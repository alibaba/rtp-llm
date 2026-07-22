import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import TensorPB


def trans_option(pb_object, py_object, name):
    if getattr(py_object, name):
        getattr(pb_object, name).value = getattr(py_object, name)


def trans_option_cast(pb_object, py_object, name, func):
    if getattr(py_object, name):
        getattr(pb_object, name).value = func(getattr(py_object, name))


def trans_grpc_dtype(type: TensorPB.DataType):
    if type == TensorPB.DataType.FP32:
        return torch.float32
    elif type == TensorPB.DataType.INT32:
        return torch.int32
    elif type == TensorPB.DataType.FP16:
        return torch.float16
    elif type == TensorPB.DataType.BF16:
        return torch.bfloat16
    else:
        raise Exception("unkown error type")


def trans_tensor(t: TensorPB):
    # Handle default-constructed (empty) TensorPB where data_type is not set.
    # trans_from_tensor returns TensorPB() for empty tensors, so data_type == 0.
    if not (len(t.shape) > 0 and t.shape[0] > 0):
        dtype = torch.float32 if t.data_type == 0 else trans_grpc_dtype(t.data_type)
        return torch.empty(list(t.shape), dtype=dtype) if t.shape else torch.empty(0, dtype=dtype)
    if t.data_type == TensorPB.DataType.FP32:
        return torch.frombuffer(t.fp32_data, dtype=torch.float32).reshape(list(t.shape))
    elif t.data_type == TensorPB.DataType.INT32:
        return torch.frombuffer(t.int32_data, dtype=torch.int32).reshape(list(t.shape))
    elif t.data_type == TensorPB.DataType.FP16:
        return torch.frombuffer(t.fp16_data, dtype=torch.float16).reshape(list(t.shape))
    elif t.data_type == TensorPB.DataType.BF16:
        return torch.frombuffer(t.bf16_data, dtype=torch.bfloat16).reshape(
            list(t.shape)
        )
    else:
        raise Exception("unkown error type")


def trans_from_tensor(t: torch.Tensor):
    if t is None:
        return TensorPB()
    res = TensorPB()
    res.shape.extend(list(t.shape))
    if t.numel() == 0:
        # Preserve shape and dtype for empty tensors; the payload stays empty.
        if t.dtype == torch.float32:
            res.data_type = TensorPB.DataType.FP32
        elif t.dtype == torch.int32:
            res.data_type = TensorPB.DataType.INT32
        elif t.dtype == torch.float16:
            res.data_type = TensorPB.DataType.FP16
        elif t.dtype == torch.bfloat16:
            res.data_type = TensorPB.DataType.BF16
        else:
            # Unsupported dtype: preserve shape, default to FP32
            res.data_type = TensorPB.DataType.FP32
            return res
        return res
    t = t.cpu()
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
