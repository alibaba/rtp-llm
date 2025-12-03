from rtp_llm.ops.compute_ops import DeviceType, get_device


def is_cuda():
    device_type = get_device().get_device_type()
    if device_type == DeviceType.Cuda:
        return True
    else:
        return False


def is_hip():
    device_type = get_device().get_device_type()
    if device_type == DeviceType.ROCm:
        return True
    else:
        return False
