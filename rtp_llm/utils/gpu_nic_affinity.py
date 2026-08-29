import json
import logging
import os
import subprocess
import tempfile
from typing import Dict, Optional


RUN_AFFINITY_PATH = "/usr/local/bin/run_affinity"
AFFINITY_OUTPUT_FILE = "npu_nic_affinity.json"


def _parse_affinity(content: str) -> Optional[Dict[str, str]]:
    try:
        affinity = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(affinity, dict) or not affinity:
        return None
    if not all(
        isinstance(rank, str) and isinstance(nic, str) and nic
        for rank, nic in affinity.items()
    ):
        return None
    return affinity


def load_gpu_nic_affinity(run_affinity_path: str = RUN_AFFINITY_PATH) -> bool:
    content = os.environ.get("ACCL_NIC_GPU_AFFINITY")
    if content is not None:
        if _parse_affinity(content) is not None:
            return True
        logging.warning("invalid ACCL_NIC_GPU_AFFINITY: %s", content)
        return False

    if not os.path.exists(run_affinity_path):
        logging.info("get gpu nic affinity failed, %s not exist", run_affinity_path)
        return False

    try:
        with tempfile.TemporaryDirectory(prefix="rtp_llm_gpu_nic_affinity_") as workdir:
            subprocess.run(
                [run_affinity_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workdir,
                timeout=30,
            )
            json_path = os.path.join(workdir, AFFINITY_OUTPUT_FILE)
            with open(json_path) as affinity_file:
                content = affinity_file.read().strip()
    except Exception as e:
        logging.info(
            "get gpu nic affinity failed, run %s failed, exception is %s",
            run_affinity_path,
            e,
        )
        return False

    affinity = _parse_affinity(content)
    if affinity is None:
        logging.warning("get gpu nic affinity failed, invalid content: %s", content)
        return False

    content = json.dumps(affinity, separators=(",", ":"))
    os.environ["ACCL_NIC_GPU_AFFINITY"] = content
    logging.info(
        "get gpu nic affinity success, set env ACCL_NIC_GPU_AFFINITY to %s",
        content,
    )
    return True


def _physical_gpu_index(local_rank: int) -> Optional[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return str(local_rank)

    devices = [device.strip() for device in visible_devices.split(",")]
    if local_rank < 0 or local_rank >= len(devices) or not devices[local_rank]:
        logging.warning(
            "local rank %d is outside CUDA_VISIBLE_DEVICES=%s",
            local_rank,
            visible_devices,
        )
        return None

    device = devices[local_rank]
    if device.isdigit():
        return device
    if not device.startswith("GPU-"):
        logging.warning("unsupported CUDA_VISIBLE_DEVICES entry: %s", device)
        return None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except Exception as e:
        logging.warning("failed to resolve CUDA device UUID %s: %s", device, e)
        return None

    for line in result.stdout.splitlines():
        index, separator, uuid = line.partition(",")
        if separator and uuid.strip().startswith(device):
            return index.strip()
    logging.warning("CUDA device UUID %s was not reported by nvidia-smi", device)
    return None


def configure_gpu_nic_affinity(local_rank: int) -> bool:
    configured_nics = os.environ.get("ACCL_USE_NICS")
    if configured_nics:
        logging.info("keep explicit ACCL_USE_NICS=%s", configured_nics)
        return True

    if not load_gpu_nic_affinity():
        return False

    content = os.environ.get("ACCL_NIC_GPU_AFFINITY", "")
    affinity = _parse_affinity(content)
    physical_gpu = _physical_gpu_index(local_rank)
    affinity_nic = (
        affinity.get(physical_gpu)
        if affinity is not None and physical_gpu is not None
        else None
    )
    # Existing PD deployments may provide a local-rank keyed mapping explicitly.
    if not affinity_nic and affinity is not None:
        affinity_nic = affinity.get(str(local_rank))
    if not affinity_nic:
        logging.warning(
            "local rank %d (physical GPU %s) get affinity nic failed, content is %s",
            local_rank,
            physical_gpu,
            content,
        )
        return False

    os.environ["ACCL_USE_NICS"] = affinity_nic
    logging.info(
        "local rank %d maps to physical GPU %s, set ACCL_USE_NICS to %s",
        local_rank,
        physical_gpu,
        affinity_nic,
    )
    return True
