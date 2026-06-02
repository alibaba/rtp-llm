"""perf_test 专用：ModelScope / HuggingFace 探测与下载（不经由 rtp_llm 主配置路径）。"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from urllib.parse import urlparse

from rtp_llm.utils.fuser import fetch_remote_file_to_local

DEFAULT_MODELSCOPE_PROBE_URL = "https://www.modelscope.cn"
DEFAULT_HUGGINGFACE_PROBE_URL = "https://huggingface.co"


def is_model_repo_id(path: str) -> bool:
    """是否为 ``org/name`` 形式的 Hub repo id（非 URL、非本地路径）。"""
    if not path or os.path.exists(os.path.expanduser(path.strip())):
        return False
    s = path.strip()
    if s.startswith(("/", ".", "~", "http://", "https://", "oss://", "hdfs://")):
        return False
    return "/" in s


def needs_perf_hub_resolve(path: str) -> bool:
    """perf 是否需要在启动引擎前将路径解析为本地（Hub 链接 / repo id / fuse 路径等）。"""
    if not path or not str(path).strip():
        return False
    expanded = os.path.expanduser(path.strip())
    if os.path.exists(expanded):
        return False
    pr = urlparse(expanded)
    if pr.scheme in ("http", "https"):
        return True
    if pr.scheme == "nas" or (pr.scheme and pr.scheme not in ("", "file")):
        return True
    if pr.scheme == "file":
        return False
    return is_model_repo_id(expanded)


def _repo_id_from_huggingface_url(url: str) -> str:
    p = urlparse(url.strip())
    parts = [x for x in p.path.split("/") if x]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    raise ValueError(f"无法从 HuggingFace URL 解析 repo id: {url}")


def _repo_id_from_modelscope_url(url: str) -> str:
    p = urlparse(url.strip())
    parts = [x for x in p.path.split("/") if x]
    if len(parts) >= 3 and parts[0] in ("models", "datasets"):
        return f"{parts[1]}/{parts[2]}"
    raise ValueError(f"无法从 ModelScope URL 解析 repo id: {url}")


def _huggingface_probe_url() -> str:
    base = (os.environ.get("HF_ENDPOINT") or DEFAULT_HUGGINGFACE_PROBE_URL).rstrip("/")
    if not base.startswith("http"):
        base = "https://" + base.lstrip("/")
    return base


def hub_reachable_with_curl(
    url: str,
    connect_timeout: int = 5,
    max_time: int = 15,
) -> bool:
    try:
        proc = subprocess.run(
            [
                "curl",
                "-sS",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "-L",
                "--connect-timeout",
                str(connect_timeout),
                "--max-time",
                str(max_time),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=max_time + connect_timeout + 5,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "未找到 curl，无法进行 ModelScope/HuggingFace 连通性检查；请安装 curl 或使用本地路径。"
        ) from e
    except subprocess.TimeoutExpired:
        return False
    if proc.returncode != 0:
        return False
    code = (proc.stdout or "").strip()
    return code.isdigit() and int(code) < 500


def assert_modelscope_http_reachable(
    connect_timeout: int = 5,
    max_time: int = 15,
) -> None:
    url = os.environ.get("RTP_MODELSCOPE_PROBE_URL", DEFAULT_MODELSCOPE_PROBE_URL)
    if not hub_reachable_with_curl(url, connect_timeout, max_time):
        raise RuntimeError(
            f"无法访问 ModelScope（curl 探测失败: {url}）。"
            f"请检查网络/代理，或使用本地路径。"
        )


def assert_huggingface_http_reachable(
    connect_timeout: int = 5,
    max_time: int = 15,
) -> None:
    url = _huggingface_probe_url()
    if not hub_reachable_with_curl(url, connect_timeout, max_time):
        raise RuntimeError(
            f"无法访问 HuggingFace Hub（curl 探测失败: {url}，已考虑 HF_ENDPOINT）。"
            f"请检查网络/代理与镜像，或使用本地路径。"
        )


from typing import Callable


class _DownloadTimeout(Exception):
    pass


def _download_with_fallback(
    ms_fn: Callable[[], str],
    hf_fn: Callable[[], str],
    tag: str,
    timeout: int,
    fallback_hint: str = "provide a local path",
) -> str:
    """Try ModelScope then HuggingFace with SIGALRM-based timeout for each attempt."""

    def _alarm_handler(signum, frame):
        raise _DownloadTimeout(f"Download timed out after {timeout}s")

    assert_modelscope_http_reachable()
    prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        try:
            signal.alarm(timeout)
            result = ms_fn()
            signal.alarm(0)
            logging.info(f"Downloaded {tag} from ModelScope: {result}")
            return result
        except _DownloadTimeout:
            logging.warning(f"ModelScope download timed out after {timeout}s for {tag}")
        except Exception as e:
            signal.alarm(0)
            logging.info(f"ModelScope download failed ({e}), trying HuggingFace")

        assert_huggingface_http_reachable()

        try:
            signal.alarm(timeout)
            result = hf_fn()
            signal.alarm(0)
            logging.info(f"Downloaded {tag} from HuggingFace: {result}")
            return result
        except _DownloadTimeout:
            raise RuntimeError(
                f"All download attempts timed out for {tag}. "
                f"Check network or {fallback_hint}."
            ) from None
        except Exception as e:
            raise RuntimeError(f"All download attempts failed for {tag}: {e}") from e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


def download_model_repo(repo_id: str, timeout: int = 600) -> str:
    def _ms():
        from modelscope import snapshot_download as ms_download

        return ms_download(repo_id)

    def _hf():
        from huggingface_hub import snapshot_download as hf_download

        return hf_download(repo_id)

    return _download_with_fallback(
        _ms,
        _hf,
        f"model '{repo_id}'",
        timeout,
        fallback_hint="provide a local --checkpoint_path",
    )


def download_dataset_repo_file(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "dataset",
    label: str = "",
    timeout: int = 300,
) -> str:
    tag = label or repo_id

    def _ms():
        from modelscope import snapshot_download as ms_download

        local_dir = ms_download(repo_id, repo_type=repo_type)
        path = os.path.join(local_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{filename} not found in downloaded repo at {local_dir}"
            )
        return path

    def _hf():
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)

    return _download_with_fallback(
        _ms,
        _hf,
        f"dataset '{tag}'",
        timeout,
        fallback_hint="use --dataset_path with a local file",
    )


def resolve_checkpoint_or_tokenizer_for_perf(path: str, timeout: int = 600) -> str:
    """将 perf 的 checkpoint/tokenizer 参数转为本地目录路径。"""
    if not path or not str(path).strip():
        raise ValueError("checkpoint/tokenizer path is empty")
    raw = path.strip()
    expanded = os.path.expanduser(raw)
    pr = urlparse(expanded)
    if pr.scheme == "nas" or (pr.scheme and pr.scheme not in ("", "file")):
        return fetch_remote_file_to_local(expanded)
    if pr.scheme in ("http", "https"):
        host = (pr.netloc or "").lower()
        if "huggingface.co" in host:
            rid = _repo_id_from_huggingface_url(expanded)
            return download_model_repo(rid, timeout=timeout)
        if "modelscope.cn" in host:
            rid = _repo_id_from_modelscope_url(expanded)
            return download_model_repo(rid, timeout=timeout)
        raise ValueError(
            f"perf_test 仅支持 huggingface.co / modelscope.cn 的 https 链接: {expanded}"
        )
    candidate = expanded
    if pr.scheme == "file":
        from urllib.request import url2pathname

        candidate = url2pathname(pr.path or "")
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)
    if os.path.isfile(candidate):
        return os.path.abspath(os.path.dirname(candidate))
    if is_model_repo_id(candidate):
        return download_model_repo(candidate, timeout=timeout)
    raise FileNotFoundError(
        f"checkpoint/tokenizer 路径不存在且非可解析的 Hub 引用: {path!r}"
    )
