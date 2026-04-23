# Multi-Node Perf Benchmark Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `rtp_llm/test/perf_test/multi_node/` fully self-contained by copying and adapting the benchmark logic from `rtp_llm/test/perf_test/` into local files, preserving the multi-node framework from the reference commit while initially targeting single-node usage.

**Architecture:** Copy 4 source files (dataclass, test_util, batch_perf_impl, batch_decode_test) into `multi_node/` as new local modules (perf_dataclass, perf_util, perf_impl, perf_runner). Copy and simplify `maga_server_manager.py` as `server_manager.py`. Refactor `local_server_runner.py` to import from these local modules and incorporate the reference commit's TCPStore coordination, logging enhancements, and multi-node endpoint discovery. No changes to any files outside `multi_node/`.

**Tech Stack:** Python 3.10, requests, psutil, prettytable, transformers, tqdm, torch.distributed.TCPStore

**Spec:** `docs/superpowers/specs/2026-04-23-multi-node-perf-benchmark-refactor-design.md`

**Reference commit:** `https://github.com/yykzjh/rtp-llm/commit/20b40a19a98dde4897cd0e090cdb344e114924e4`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `rtp_llm/test/perf_test/multi_node/perf_dataclass.py` | Create | Data classes: `ResponseInfo`, `TestResultMetrics`, `analyze_results()`, `MetricState`, `TableType`, `create_metrics_table()` |
| `rtp_llm/test/perf_test/multi_node/perf_util.py` | Create | Tokenizer loading and query generation: `_load_tokenizer()`, `get_prompt()`, `create_query()` |
| `rtp_llm/test/perf_test/multi_node/perf_impl.py` | Create | Benchmark execution engine: `BatchPerfImpl` with multi-node endpoint discovery |
| `rtp_llm/test/perf_test/multi_node/perf_runner.py` | Create | Benchmark orchestrator: `run_single()` function |
| `rtp_llm/test/perf_test/multi_node/server_manager.py` | Create | Server lifecycle management: `LocalServerManager` (simplified from `MagaServerManager`) |
| `rtp_llm/test/perf_test/multi_node/local_server_runner.py` | Modify | Main entry: rewire imports to local modules, add TCPStore coordination, logging enhancements |

---

### Task 1: Create `perf_dataclass.py`

**Files:**
- Create: `rtp_llm/test/perf_test/multi_node/perf_dataclass.py`

This is a direct copy of `rtp_llm/test/perf_test/dataclass.py` — the contents are identical. It has no dependency on other perf_test files.

- [ ] **Step 1: Create `perf_dataclass.py`**

Copy the entire contents of `rtp_llm/test/perf_test/dataclass.py` into `rtp_llm/test/perf_test/multi_node/perf_dataclass.py`. The file contains:

```python
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from prettytable import PrettyTable


class ResponseInfo:
    success: bool = False
    input_len: int = 0
    output_len: int = 0
    wait_time: float = 0.0
    total_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    decode_time_per_token: float = 0.0

    def __init__(self, response: dict, success: bool = True):
        if not success:
            return
        self.success = success
        aux_info = response.get("aux_info", {})
        self.input_len = aux_info.get("input_len", 0)
        self.output_len = aux_info.get("output_len", 0)
        self.wait_time = aux_info.get("wait_time", 0.0)
        self.total_time = aux_info.get("cost_time", 0.0) - self.wait_time
        self.prefill_time = aux_info.get("first_token_cost_time", 0.0) - self.wait_time
        self.decode_time = self.total_time - self.prefill_time
        self.decode_time_per_token = (
            self.decode_time / (self.output_len - 1) if self.output_len > 1 else 0.0
        )


@dataclass
class TestResultMetrics:
    total_requests: int
    success_requests: int
    fail_requests: int
    avg_input_len: float = 0.0
    avg_output_len: float = 0.0
    avg_wait_time: float = 0.0
    max_wait_time: float = 0.0
    avg_total_time: float = 0.0
    max_total_time: float = 0.0
    avg_prefill_time: float = 0.0
    max_prefill_time: float = 0.0
    prefill_time_var: float = 0.0
    avg_decode_time: float = 0.0
    max_decode_time: float = 0.0
    decode_time_var: float = 0.0


def analyze_results(responses: List[ResponseInfo]) -> TestResultMetrics:
    total_request_count = len(responses)
    success_requests = [r for r in responses if r.success]
    success_count = len(success_requests)
    fail_count = total_request_count - success_count
    metrics = TestResultMetrics(
        total_requests=total_request_count,
        success_requests=success_count,
        fail_requests=fail_count,
    )
    if success_count:
        metrics.avg_input_len = (
            sum([r.input_len for r in success_requests]) / success_count
        )
        metrics.avg_output_len = (
            sum([r.output_len for r in success_requests]) / success_count
        )
        metrics.avg_wait_time = (
            sum([r.wait_time for r in success_requests]) / success_count
        )
        metrics.max_wait_time = max([r.wait_time for r in success_requests])
        metrics.avg_total_time = (
            sum([r.total_time for r in success_requests]) / success_count
        )
        metrics.max_total_time = max([r.total_time for r in success_requests])
        metrics.avg_prefill_time = (
            sum([r.prefill_time for r in success_requests]) / success_count
        )
        metrics.max_prefill_time = max([r.prefill_time for r in success_requests])
        metrics.prefill_time_var = (
            sum(
                [
                    (r.prefill_time - metrics.avg_prefill_time) ** 2
                    for r in success_requests
                ]
            )
            / success_count
        )
        metrics.avg_decode_time = (
            sum([r.decode_time_per_token for r in success_requests]) / success_count
        )
        metrics.max_decode_time = max(
            [r.decode_time_per_token for r in success_requests]
        )
        metrics.decode_time_var = (
            sum(
                [
                    (r.decode_time_per_token - metrics.avg_decode_time) ** 2
                    for r in success_requests
                ]
            )
            / success_count
        )
    return metrics


class MetricState(object):
    def __init__(self, input_len: int, batch_size: int, metrics: TestResultMetrics):
        self.input_len = input_len
        self.batch_size = batch_size
        self.metrics = metrics


class TableType(Enum):
    Prefill = "prefill"
    Decode = "decode"


def create_metrics_table(
    table_type: TableType,
    metrics_list: List[MetricState],
    dump_json_path: str,
    model_info: Dict[str, Any],
    title: str,
    generate_config: Dict[str, Any] = {},
) -> str:
    json_result: Dict[str, Any] = {
        "title": title,
        "metrics": [],
        "model_info": model_info,
        "generate_config": generate_config,
    }
    main_table = PrettyTable()
    main_table.title = title
    main_table.field_names = [
        "Seq Len",
        "Batch Size",
        "Sucess/Total Req",
        "Input/Output",
        "Waiting Time(ms)",
    ] + (
        ["Prefill Time(ms)"] if table_type == TableType.Prefill else ["Decode Time(ms)"]
    )
    for metrics_item in metrics_list:
        metrics = metrics_item.metrics
        if metrics.success_requests > 0:
            main_table.add_row(
                [
                    metrics_item.input_len,
                    metrics_item.batch_size,
                    f"{metrics.success_requests}/{metrics.total_requests}",
                    f"{metrics.avg_input_len:.0f}/{metrics.avg_output_len:.0f}",
                    f"{metrics.avg_wait_time:.2f}",
                ]
                + (
                    [f"{metrics.avg_prefill_time:.2f}"]
                    if table_type == TableType.Prefill
                    else [f"{metrics.avg_decode_time:.2f}"]
                )
            )
            json_result["metrics"].append(
                {
                    "input_len": metrics_item.input_len,
                    "batch_size": metrics_item.batch_size,
                    "success_rate": metrics.success_requests / metrics.total_requests,
                    "avg_wait_time": metrics.avg_wait_time,
                    "avg_prefill_time": metrics.avg_prefill_time,
                    "avg_decode_time": metrics.avg_decode_time,
                }
            )
        else:
            main_table.add_row(
                [
                    metrics_item.input_len,
                    metrics_item.batch_size,
                    f"0/{metrics.total_requests}",
                    "N/A",
                    "N/A",
                    "N/A",
                ]
            )
    os.makedirs(dump_json_path, exist_ok=True)
    with open(f"{dump_json_path}/{title.replace(' ', '_')}.json", "w") as f:
        json.dump(json_result, f, indent=4)
    main_table.align = "l"
    return main_table.get_string()
```

- [ ] **Step 2: Verify the file was created correctly**

Run:
```bash
python -c "from rtp_llm.test.perf_test.multi_node.perf_dataclass import ResponseInfo, TestResultMetrics, analyze_results, MetricState, TableType, create_metrics_table; print('All imports OK')"
```
Expected: `All imports OK`

- [ ] **Step 3: Commit**

```bash
git add rtp_llm/test/perf_test/multi_node/perf_dataclass.py
git commit -m "feat(multi_node): add perf_dataclass module copied from dataclass.py"
```

---

### Task 2: Create `perf_util.py`

**Files:**
- Create: `rtp_llm/test/perf_test/multi_node/perf_util.py`

Copy from `rtp_llm/test/perf_test/test_util.py` but **remove** the `write_odps()` function and the `odps` import. Keep only tokenizer loading and query generation.

- [ ] **Step 1: Create `perf_util.py`**

```python
import os
from typing import Any, Dict, List

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rtp_llm.utils.fuser import fetch_remote_file_to_local


def _load_tokenizer(model_type: str, tokenizer_path: str) -> PreTrainedTokenizerBase:
    """Load tokenizer, with GLM-5 compatible path (bypass invalid tokenizer_config.json)."""
    if model_type == "glm_5":
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(
                f"GLM-5 tokenizer requires tokenizer.json at {tokenizer_file}"
            )
        tokenizer = Tokenizer.from_file(tokenizer_file)
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            unk_token="<|endoftext|>",
        )
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def get_prompt(tokenizer: Any, prompt: str, seqlen: int):
    while len(tokenizer.encode(prompt)) < seqlen:
        prompt += prompt
    for dec_step in [1024, 256, 64, 16, 2, 1]:
        while len(tokenizer.encode(prompt[:-dec_step])) >= seqlen:
            prompt = prompt[:-dec_step]
    return prompt


def create_query(
    model_type: str, tokenizer_path: str, input_len_list: List[int]
) -> Dict[int, str]:
    tokenizer_path = fetch_remote_file_to_local(tokenizer_path)

    def _create_query_single(tokenizer: PreTrainedTokenizerBase, input_len: int) -> str:
        base_query = "hello " * (input_len + 20)

        def get_token_length(text: str) -> int:
            return len(tokenizer.encode(text))

        left, right = 0, len(base_query)
        while left < right:
            mid = (left + right) // 2
            current_query = base_query[:mid]
            current_len = get_token_length(current_query)
            if current_len == input_len:
                return current_query
            elif current_len < input_len:
                left = mid + 1
            else:
                right = mid
        return base_query[:left]

    tokenizer = _load_tokenizer(model_type, tokenizer_path)
    return {x: _create_query_single(tokenizer, x) for x in input_len_list}
```

- [ ] **Step 2: Verify imports**

Run:
```bash
python -c "from rtp_llm.test.perf_test.multi_node.perf_util import create_query, get_prompt, _load_tokenizer; print('All imports OK')"
```
Expected: `All imports OK`

- [ ] **Step 3: Commit**

```bash
git add rtp_llm/test/perf_test/multi_node/perf_util.py
git commit -m "feat(multi_node): add perf_util module from test_util.py without ODPS"
```

---

### Task 3: Create `perf_impl.py`

**Files:**
- Create: `rtp_llm/test/perf_test/multi_node/perf_impl.py`

This is the reference commit version of `batch_perf_impl.py` with multi-node endpoint discovery. Key changes from the current `batch_perf_impl.py`:
- `_curl_server_single_worker()` uses `tp0_endpoints` for host/port instead of hardcoded `127.0.0.1`
- `BatchPerfImpl` constructor accepts `gang_config_string`, `local_world_size`, `request_tpot`, `connection_timeout`, `retry_times`, `retry_interval`
- `_get_all_dp_tp0_frontends()` discovers endpoints from `gang_config_string`
- `_set_concurrency()` uses multi-threaded POST with retry
- Import from local `perf_dataclass` instead of `rtp_llm.test.perf_test.dataclass`

- [ ] **Step 1: Create `perf_impl.py`**

```python
import json
import logging
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import requests

from rtp_llm.distribute.distributed_server import members_from_test_env
from rtp_llm.test.perf_test.multi_node.perf_dataclass import (
    ResponseInfo,
    TestResultMetrics,
    analyze_results,
)
from rtp_llm.utils.util import check_with_info


def _curl_server_single_worker(
    request_id: int,
    batch_size: int,
    tp0_endpoints: List[Tuple[str, int]],
    input_query: str,
    is_decode: bool,
    decode_test_length: int,
    request_timeout: int,
    is_profile: bool = False,
    is_warmup: bool = False,
    generate_config: Dict[str, Any] = {},
) -> ResponseInfo:
    """Curl the server for a single request"""
    batch_idx = request_id // batch_size
    host = tp0_endpoints[batch_idx][0]
    port = tp0_endpoints[batch_idx][1]
    req = {
        "prompt": input_query,
        "generate_config": {
            "max_new_tokens": decode_test_length if is_decode else 1,
            "min_new_tokens": decode_test_length if is_decode else 1,
            "force_sp_accept": True,
        },
    }

    if generate_config:
        req["generate_config"].update(generate_config)
        if "top_k" in generate_config:
            req["top_k"] = generate_config["top_k"]
        if "top_p" in generate_config:
            req["top_p"] = generate_config["top_p"]

    if "top_k" not in req:
        req["top_k"] = 1

    if is_profile:
        req["gen_timeline"] = True
        req["profile_step"] = 1

    if is_warmup:
        request_timeout = 1000

    try:
        response = requests.post(
            f"http://{host}:{port}", json=req, timeout=request_timeout
        )
        if response.status_code != 200:
            logging.warning(f"request failed: {response.content}")
            return ResponseInfo({}, False)
        logging.debug(response.text)
        return ResponseInfo(response.json())
    except Exception as e:
        logging.warning(f" request exception: {e}")
        return ResponseInfo({}, False)


def _curl_server_batch_worker(
    request_indices: List[int],
    batch_size: int,
    tp0_endpoints: List[Tuple[str, int]],
    input_query: str,
    is_decode: bool,
    decode_test_length: int,
    request_timeout: int,
    is_profile: bool = False,
    is_warmup: bool = False,
    generate_config: Dict[str, Any] = {},
) -> List[ResponseInfo]:
    """Use ThreadPoolExecutor to concurrently handle multiple requests"""
    responses = []

    with ThreadPoolExecutor(max_workers=len(request_indices)) as executor:
        futures = []
        for request_id in request_indices:
            future = executor.submit(
                _curl_server_single_worker,
                request_id,
                batch_size,
                tp0_endpoints,
                input_query,
                is_decode,
                decode_test_length,
                request_timeout,
                is_profile,
                is_warmup,
                generate_config,
            )
            futures.append(future)

        for future in futures:
            response = future.result()
            responses.append(response)

    return responses


class BatchPerfImpl(object):
    def __init__(
        self,
        base_port: int,
        dp_size: int,
        tp_size: int,
        local_world_size: int,
        batch_size: int,
        input_len: int,
        query: str,
        gang_config_string: str,
        request_tpot: int = 100,
        connection_timeout: int = 10,
        retry_times: int = 3,
        retry_interval: float = 0.5,
        is_decode: bool = True,
        decode_test_length: int = 10,
        generate_config: Dict[str, Any] = {},
    ):
        self.base_port = base_port
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.local_world_size = local_world_size
        self.total_batch_size = batch_size
        self.input_len = input_len
        self.input_query = query
        self.is_decode = is_decode
        self.max_requests_per_process = 128
        self.gang_config_string = gang_config_string
        self.connection_timeout = connection_timeout
        self.retry_times = retry_times
        self.retry_interval = retry_interval
        self.request_timeout = (
            30 + request_tpot * decode_test_length // 1000 + connection_timeout
        )
        self.num_processes = max(
            1,
            (self.total_batch_size + self.max_requests_per_process - 1)
            // self.max_requests_per_process,
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.executor = ProcessPoolExecutor(max_workers=self.num_processes)
        self.decode_test_length = decode_test_length
        self.generate_config = generate_config
        self.tp0_endpoints = self._get_all_dp_tp0_frontends()
        logging.info(f"tp0_endpoints: {self.tp0_endpoints}")

    def run(self):
        self._set_concurrency()
        logging.info(f"finished setting concurrency")
        _ = self._curl_server(is_profile=False, is_warmup=True)
        logging.info(f"finished warmup")
        results = self._curl_server(is_profile=False, is_warmup=False)
        logging.info(f"finished measure time")
        _ = self._curl_server(is_profile=True, is_warmup=False)
        logging.info(f"finished dump profile json")
        return results

    def _set_concurrency(self):
        check_with_info(
            self.total_batch_size % self.dp_size == 0,
            f"concurrency {self.total_batch_size} must be divisible by dp_size {self.dp_size}",
        )
        batch_size = self.total_batch_size // self.dp_size

        payload = {
            "batch_size": batch_size,
            "mode": "decode" if self.is_decode else "prefill",
        }

        def _post_one(ip: str, port: int) -> Tuple[str, int, int, str]:
            retry_count = 0
            url = f"http://{ip}:{port}/update_scheduler_info"
            resp_status_code = 0
            resp_status_text = "not ok"
            while retry_count < self.retry_times:
                time.sleep(self.retry_interval)
                resp = requests.post(url, json=payload, timeout=self.connection_timeout)
                resp_status_code = resp.status_code
                resp_status_text = resp.json().get("status", "not ok")
                if resp_status_code == 200 and resp_status_text == "ok":
                    break
                else:
                    retry_count += 1
                    logging.warning(
                        f"update_scheduler_info request failed for {ip}:{port}: status code={resp_status_code}, status text={resp_status_text}, retry_count={retry_count}/{self.retry_times}"
                    )
            return ip, port, resp_status_code, resp_status_text

        max_workers = min(len(self.tp0_endpoints), 32) if self.tp0_endpoints else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_post_one, ip, port)
                for ip, port in reversed(self.tp0_endpoints)
            ]

        errors: List[str] = []
        for fut in futures:
            try:
                ip, port, status_code, status_text = fut.result()
                if status_code != 200 or status_text != "ok":
                    errors.append(
                        f"update_scheduler_info failed for {ip}:{port}: status={status_code}, response={status_text}"
                    )
            except Exception as e:
                errors.append(f"update_scheduler_info exception: {e}")

        if errors:
            raise Exception(";\n".join(errors))

    def _get_all_dp_tp0_frontends(self) -> List[Tuple[str, int]]:
        """Get all DP groups' tp_rank==0 frontend endpoints across nodes from gang_config_string."""
        nodes = members_from_test_env(self.gang_config_string)
        targets: List[Tuple[str, int]] = []
        for dp_rank in range(self.dp_size):
            tp0_world_rank = dp_rank * self.tp_size
            node_idx = tp0_world_rank // self.local_world_size
            local_rank = tp0_world_rank % self.local_world_size
            check_with_info(
                node_idx < len(nodes),
                f"dp_rank {dp_rank} (tp0_world_rank={tp0_world_rank}) maps to node_idx={node_idx}, "
                f"but only {len(nodes)} nodes in GANG_CONFIG_STRING",
            )
            base_port = int(nodes[node_idx].server_port)
            port = base_port + local_rank * int(
                os.environ.get("WORKER_INFO_PORT_NUM", "10")
            )
            targets.append((nodes[node_idx].ip, int(port)))
        return list(dict.fromkeys(targets))

    def _curl_server(
        self, is_profile: bool = False, is_warmup: bool = False
    ) -> TestResultMetrics:
        request_batches: List[List[int]] = []
        for i in range(0, self.total_batch_size, self.max_requests_per_process):
            batch_indices = list(
                range(i, min(i + self.max_requests_per_process, self.total_batch_size))
            )
            request_batches.append(batch_indices)

        futures: List[Future[List[ResponseInfo]]] = []
        for batch_indices in request_batches:
            futures.append(
                self.executor.submit(
                    _curl_server_batch_worker,
                    batch_indices,
                    self.total_batch_size // self.dp_size,
                    self.tp0_endpoints,
                    self.input_query,
                    self.is_decode,
                    self.decode_test_length,
                    self.request_timeout,
                    is_profile,
                    is_warmup,
                    self.generate_config,
                )
            )

        all_responses: List[ResponseInfo] = []
        for future in futures:
            batch_responses = future.result()
            all_responses.extend(batch_responses)

        metrics = analyze_results(all_responses)
        return metrics

    def dump_results(self, results: List[Dict[str, Any]]):
        for result in results:
            logging.debug(json.dumps(result))
```

- [ ] **Step 2: Verify imports**

Run:
```bash
python -c "from rtp_llm.test.perf_test.multi_node.perf_impl import BatchPerfImpl; print('All imports OK')"
```
Expected: `All imports OK`

- [ ] **Step 3: Commit**

```bash
git add rtp_llm/test/perf_test/multi_node/perf_impl.py
git commit -m "feat(multi_node): add perf_impl module with multi-node endpoint discovery"
```

---

### Task 4: Create `perf_runner.py`

**Files:**
- Create: `rtp_llm/test/perf_test/multi_node/perf_runner.py`

Extract only `run_single()` from the reference commit version of `batch_decode_test.py`. Key differences from current code:
- Accepts `gang_config_string`, `local_world_size`, `request_tpot`, `connection_timeout`, `retry_times`, `retry_interval`
- Uses named arguments when constructing `BatchPerfImpl`
- No warmup pass (warmup is now inside `BatchPerfImpl.run()`)
- Uses `logging.info` instead of `pbar.set_description`
- Imports from local `perf_impl` and `perf_dataclass`

- [ ] **Step 1: Create `perf_runner.py`**

```python
import logging
import os
from typing import Any, Dict, List, Optional

from rtp_llm.test.perf_test.multi_node.perf_dataclass import (
    MetricState,
    TableType,
    create_metrics_table,
)
from rtp_llm.test.perf_test.multi_node.perf_impl import BatchPerfImpl

from tqdm import tqdm


def run_single(
    base_port: int,
    dp_size: int,
    tp_size: int,
    batch_size_list: List[int],
    input_len_list: List[int],
    input_query_dict: Dict[int, str],
    gang_config_string: Optional[str] = None,
    local_world_size: int = 0,
    request_tpot: int = 100,
    connection_timeout: int = 10,
    retry_times: int = 3,
    retry_interval: float = 0.5,
    is_decode: bool = True,
    dump_json_path: str = ".",
    decode_test_length: int = 10,
    is_speculative: bool = False,
    propose_step: int = 0,
    generate_config: Dict[str, Any] = {},
) -> List[MetricState]:
    if not local_world_size:
        local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE", str(dp_size * tp_size))
        )
    if not gang_config_string:
        gang_config_string = os.environ.get("GANG_CONFIG_STRING", "")
    if not gang_config_string:
        gang_config_string = f"name:perf_part0,ip:127.0.0.1,port:{base_port}"

    title_prefix = f"Speculative(step={propose_step}) " if is_speculative else ""
    title = "Decode Result" if is_decode else "Prefill Result"
    title = f"{title_prefix}{title}"
    batch_size_list = [1] if not is_decode else batch_size_list

    metrics_list: List[MetricState] = []

    total_tests = len(batch_size_list) * len(input_len_list)
    with tqdm(total=total_tests, desc=f"Running {title}", unit="test") as pbar:
        for batch_size in batch_size_list:
            for input_len in input_len_list:
                logging.info(
                    f"Running {title} - batch_size: {batch_size}, input_len: {input_len}"
                )
                metric = BatchPerfImpl(
                    base_port=base_port,
                    dp_size=dp_size,
                    tp_size=tp_size,
                    local_world_size=local_world_size,
                    batch_size=batch_size * dp_size,
                    input_len=input_len,
                    query=input_query_dict[input_len],
                    gang_config_string=gang_config_string,
                    request_tpot=request_tpot,
                    connection_timeout=connection_timeout,
                    retry_times=retry_times,
                    retry_interval=retry_interval,
                    is_decode=is_decode,
                    decode_test_length=decode_test_length,
                    generate_config=generate_config,
                ).run()
                metrics_list.append(MetricState(input_len, batch_size, metric))

                pbar.update(1)

    metrics_table = create_metrics_table(
        TableType.Decode if is_decode else TableType.Prefill,
        metrics_list,
        dump_json_path,
        {"dp_size": dp_size, "tp_size": tp_size},
        title,
        generate_config,
    )
    logging.info("metrics_table: \n" + str(metrics_table))
    return metrics_list
```

- [ ] **Step 2: Verify imports**

Run:
```bash
python -c "from rtp_llm.test.perf_test.multi_node.perf_runner import run_single; print('All imports OK')"
```
Expected: `All imports OK`

- [ ] **Step 3: Commit**

```bash
git add rtp_llm/test/perf_test/multi_node/perf_runner.py
git commit -m "feat(multi_node): add perf_runner module with run_single function"
```

---

### Task 5: Create `server_manager.py`

**Files:**
- Create: `rtp_llm/test/perf_test/multi_node/server_manager.py`

Simplified copy of `rtp_llm/test/utils/maga_server_manager.py`. Removed: `get_free_port()`, `PortManager`, `visit()`, `smoke_args_str`, `_role_name`, `_device_ids`, `MIN_WORKER_INFO_PORT_NUM`. The class is renamed to `LocalServerManager`. The `wait_sever_done()` method is inlined (calls the same utility from `rtp_llm.utils.util`). The `start_server()` signature is simplified — it reads model config from environment variables only (no `model_path`/`model_type`/`tokenizer_path` parameters needed since they're already in env).

- [ ] **Step 1: Create `server_manager.py`**

```python
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import psutil
import requests


class LocalServerManager(object):
    def __init__(
        self,
        port: int,
        log_dir: str,
        process_file_name: str = "process.log",
    ):
        self._port = port
        self._log_dir = log_dir
        self._process_file_name = process_file_name
        self._log_file = None
        self._file_stream = None
        self._server_process = None

    def __del__(self):
        self.stop_server()

    @property
    def port(self) -> int:
        return self._port

    def start_server(
        self,
        retry_interval: int = 1,
        check_connection_timeout: int = 10,
        timeout: int = 1600,
    ) -> bool:
        role_log_name = "main_logs"
        bazel_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
        cwd_path = os.environ.get("MAGA_SERVER_WORK_DIR", bazel_outputs_dir)

        current_env: Dict[str, str] = os.environ.copy()
        current_env["START_PORT"] = str(self._port)

        if "DG_JIT_CACHE_DIR" not in current_env:
            home_dir = os.environ.get("HOME", os.path.expanduser("~"))
            current_env["DG_JIT_CACHE_DIR"] = os.path.join(home_dir, ".deep_gemm")

        log_dir = os.path.join(bazel_outputs_dir, role_log_name)
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = os.path.join(log_dir, self._process_file_name)
        self._file_stream = open(self._log_file, "w")

        logging.info(f"log file: {self._log_file}")

        p = subprocess.Popen(
            ["/opt/conda310/bin/python", "-m", "rtp_llm.start_server"],
            env=current_env,
            stdout=self._file_stream,
            stderr=self._file_stream,
            cwd=cwd_path,
        )
        self._server_process = p

        return self._wait_server_done(
            retry_interval=retry_interval,
            timeout=timeout,
        )

    def _wait_server_done(
        self,
        retry_interval: int = 1,
        timeout: int = 1600,
    ) -> bool:
        host = "localhost"
        port = str(self._port)
        start_time = time.time()

        logging.info(f"waiting for pid[{self._server_process.pid}] to start on port {port}")
        while True:
            try:
                response = requests.get(
                    f"http://{host}:{port}/health", timeout=retry_interval
                )
                logging.info(
                    f"response status_code = {response.status_code}, text = {response.text}"
                )
                if response.status_code == 200:
                    logging.info(f"port {port} started successfully")
                    return True
                else:
                    logging.debug(f"health check is not ready")
            except BaseException as e:
                logging.debug("health check is not ready, %s", str(e))

            time.sleep(retry_interval)

            if not psutil.pid_exists(self._server_process.pid) or self._server_process.poll():
                logging.warning(
                    f"process:[{self._server_process.pid}] status abnormal, server start failed"
                )
                return False
            if time.time() - start_time > timeout:
                logging.warning(f"waiting for port {port} startup timeout")
                return False

    def stop_server(self):
        if self._server_process is not None and self._server_process.pid is not None:
            try:
                logging.info("stop server and children: %d", self._server_process.pid)
                parent = psutil.Process(self._server_process.pid)
                children = list(parent.children(recursive=True))
                for child in children:
                    child.terminate()
                _, alive = psutil.wait_procs(children, timeout=5)
                for child in alive:
                    child.kill()
                parent.terminate()
                try:
                    parent.wait(timeout=10)
                except psutil.TimeoutExpired:
                    logging.warning(
                        "Parent process did not exit gracefully, force killing"
                    )
                    parent.kill()
                    parent.wait(timeout=5)
                self._server_process = None
            except Exception as e:
                logging.warning("failed to get process with: " + str(e))
                self._server_process = None
        if self._file_stream is not None:
            self._file_stream.close()
            self._file_stream = None
        return True

    def print_process_log(self):
        if self._log_file is None:
            return
        if self._file_stream is not None:
            try:
                self._file_stream.flush()
            except Exception:
                pass
        try:
            if os.path.exists(self._log_file):
                with open(self._log_file, "r") as f:
                    content = f.read()
                if content:
                    logging.warning("=" * 80)
                    logging.warning(f"Server process log ({self._log_file}):")
                    logging.warning("=" * 80)
                    logging.warning(f"{content}")
                    logging.warning("=" * 80)
                else:
                    logging.warning(f"Log file {self._log_file} is empty")
            else:
                logging.warning(f"Log file {self._log_file} does not exist")
        except Exception as e:
            logging.warning(f"Failed to read log file {self._log_file}: {e}")
```

- [ ] **Step 2: Verify imports**

Run:
```bash
python -c "from rtp_llm.test.perf_test.multi_node.server_manager import LocalServerManager; print('All imports OK')"
```
Expected: `All imports OK`

- [ ] **Step 3: Commit**

```bash
git add rtp_llm/test/perf_test/multi_node/server_manager.py
git commit -m "feat(multi_node): add server_manager module simplified from maga_server_manager"
```

---

### Task 6: Refactor `local_server_runner.py`

**Files:**
- Modify: `rtp_llm/test/perf_test/multi_node/local_server_runner.py`

This is the biggest change. Rewrite the entire file following the reference commit, but with imports from local modules instead of `rtp_llm.test.perf_test.*`. Key changes:

1. Import from local `perf_runner`, `perf_util`, `server_manager`
2. Import `LocalServerManager` instead of `MagaServerManager`
3. Add `patch_logging_stream_handler()`
4. Add TCPStore coordination functions (`_init_startup_store`, `_store_set_safe`, `_store_check_failed`, `_store_check_ok`, `_store_barrier`)
5. Add `wait_world_server_startup()`
6. Enhanced `wait_master_done()` with configurable parameters
7. Remove `test_main()` wrapper
8. Restructure `__main__` block per reference commit

- [ ] **Step 1: Rewrite `local_server_runner.py`**

Replace the entire file with:

```python
import os

for key, value in os.environ.items():
    print(f"start env {key}={value}")

import json
import logging
import pathlib
import signal
import socket
import sys
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional

import requests

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.parent.parent.absolute()))
from rtp_llm.utils.import_util import has_internal_source

if has_internal_source():
    from internal_source.rtp_llm.test.util.set_internal_env import (
        configure_optional_env,
    )

    configure_optional_env()

import torch.distributed as dist
from torch.distributed import TCPStore

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.distribute.distributed_server import members_from_test_env
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.test.perf_test.multi_node.perf_runner import run_single
from rtp_llm.test.perf_test.multi_node.perf_util import create_query
from rtp_llm.test.perf_test.multi_node.server_manager import LocalServerManager

# from uvicorn.loops.uvloop import uvloop_setup
# uvloop_setup()

_SERVER_STARTUP_STORE_KEY_FAILED = "server_startup_failed"
_SERVER_STARTUP_STORE_KEY_OK_PREFIX = "server_startup_ok_"
_SERVER_STARTUP_BARRIER_PREFIX = "barrier/"
_SERVER_STARTUP_BARRIER_RELEASE_KEY = "barrier/release"


def patch_logging_stream_handler():
    """
    Add a StreamHandler to stdout to ensure logs can be captured by subprocess.run to test.log.
    """
    root_logger = logging.getLogger()

    has_console_stream_handler = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)
            if stream in (sys.stdout, sys.stderr):
                has_console_stream_handler = True
                break

    if has_console_stream_handler:
        logging.debug("Console StreamHandler already exists, skipping")
        return False

    try:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(
            root_logger.level if root_logger.level else logging.INFO
        )
        formatter = logging.Formatter(
            "[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
        logging.debug("Successfully added StreamHandler to stdout")
        return True
    except Exception as e:
        logging.warning(f"Failed to add StreamHandler to stdout: {e}")
        return False


def _init_startup_store(
    py_env_configs: PyEnvConfigs,
    world_rank: int,
    timeout: int,
) -> Optional[TCPStore]:
    """
    A lightweight cross-node coordination channel for perf tests.

    We intentionally DO NOT reuse the server's own TCPStore port (master_port-1),
    because the real distributed server will also bind it. Use master_port-12 instead.
    """
    try:
        dist_config_str = os.environ.get(
            "GANG_CONFIG_STRING", py_env_configs.distribute_config.gang_config_string
        )
        if not dist_config_str:
            raise RuntimeError(
                "no gang config string (GANG_CONFIG_STRING), unexpected!"
            )
        dist_members = members_from_test_env(dist_config_str)
        master_member = dist_members[0]
        master_ip = master_member.ip
        master_port = int(master_member.server_port)
        store_port = master_port - 12
        if store_port <= 0:
            raise RuntimeError(f"invalid startup store port: {store_port}")

        store_timeout = timedelta(seconds=timeout)
        logging.info(
            f"init startup store {master_ip}:{store_port}, world_size={len(dist_members)}, world_rank={world_rank}"
        )
        store = dist.TCPStore(
            host_name=master_ip,
            port=store_port,
            world_size=len(dist_members),
            is_master=(world_rank == 0),
            wait_for_workers=False,
            timeout=store_timeout,
        )
        return store
    except Exception as e:
        logging.warning(
            f"failed to init startup store, fallback to timeout wait. err={e}"
        )
        return None


def _store_set_safe(store: Optional[Any], key: str, value: str) -> None:
    if store is None:
        return
    try:
        store.set(key, value)
    except Exception as e:
        logging.warning(f"startup store set failed, key={key}, err={e}")


def _store_check_failed(store: Optional[Any]) -> Optional[str]:
    if store is None:
        return None
    try:
        if store.check([_SERVER_STARTUP_STORE_KEY_FAILED]):
            v = store.get(_SERVER_STARTUP_STORE_KEY_FAILED)
            try:
                return str(v, encoding="utf-8")
            except Exception:
                return str(v)
    except Exception:
        return None
    return None


def _store_check_ok(store: Optional[Any], key: str) -> bool:
    if store is None:
        return False
    try:
        v = store.get(key)
        v_str = str(v, encoding="utf-8")
        logging.info(f"store key {key} value bytes={v}, value str={v_str}")
        return v_str == "ok"
    except Exception:
        return False


def _store_barrier(store: TCPStore, node_rank: int, node_world_size: int) -> None:
    """
    A pure TCPStore-based barrier (no dist.init_process_group required).
    """
    try:
        store.set(f"{_SERVER_STARTUP_BARRIER_PREFIX}{node_rank}", "1")
    except Exception as e:
        logging.warning(f"store barrier set failed, node_rank={node_rank}, err={e}")
        raise

    if node_rank == 0:
        store.wait(
            [f"{_SERVER_STARTUP_BARRIER_PREFIX}{i}" for i in range(node_world_size)]
        )
        store.set(_SERVER_STARTUP_BARRIER_RELEASE_KEY, "1")
    else:
        store.wait([_SERVER_STARTUP_BARRIER_RELEASE_KEY])


def wait_master_done(
    env_dict: Dict[str, str] = {},
    world_rank: int = 0,
    py_env_configs: PyEnvConfigs = None,
    retry_interval: int = 1,
    retry_times: int = 3,
    heartbeat_interval: int = 10,
    check_connection_timeout: int = 10,
) -> None:
    dist_config_str = env_dict.get(
        "GANG_CONFIG_STRING", py_env_configs.distribute_config.gang_config_string
    )
    if not dist_config_str:
        raise RuntimeError("no gang config string, unexpected!")
    dist_members = members_from_test_env(dist_config_str)
    master_member = dist_members[0]
    master_host = master_member.ip
    master_port = master_member.server_port
    while True:
        logging.info(
            f"rank [{world_rank}] waiting for master {master_host}:{master_port} done"
        )
        time.sleep(heartbeat_interval)

        retry_count = 0
        connection_failed = True
        while True:
            try:
                sock = socket.create_connection(
                    (master_host, master_port), timeout=check_connection_timeout
                )
                sock.close()
                connection_failed = False
                break
            except (socket.error, ConnectionRefusedError) as e:
                retry_count += 1
                if retry_count >= retry_times:
                    break
                logging.info(
                    f"rank [{world_rank}] connection attempt {retry_count} failed, retrying... Error: {e}"
                )
                time.sleep(retry_interval)

        if connection_failed:
            break

    logging.info(
        f"rank [{world_rank}] master {master_host}:{master_port} done, worker exit!"
    )
    return


def wait_world_server_startup(
    tcp_store: TCPStore,
    py_env_configs: PyEnvConfigs,
    check_interval: int = 3,
    check_connection_timeout: int = 10,
    server_startup_timeout: int = 1600,
):
    """
    Wait for all servers to complete startup.
    """
    start_time = time.time()

    dist_config_str = py_env_configs.distribute_config.gang_config_string
    if not dist_config_str:
        raise RuntimeError("no gang config string, unexpected!")
    dist_members = members_from_test_env(dist_config_str)
    targets = [(m.name, m.ip, int(m.server_port)) for m in dist_members]
    logging.info(
        f"waiting all servers startup, targets={targets}, check_interval={check_interval}s, check_connection_timeout={check_connection_timeout}s, server_startup_timeout={server_startup_timeout}s"
    )

    def _is_ready(node_rank: int, host: str, port: int) -> bool:
        try:
            health_resp = requests.get(
                f"http://{host}:{port}/health", timeout=check_connection_timeout
            )
            health_resp_status_code = health_resp.status_code
            health_resp_status_text = health_resp.json()
            health_ok = (
                health_resp_status_code == 200 and health_resp_status_text == "ok"
            )
            update_scheduler_info_resp = requests.post(
                f"http://{host}:{port}/update_scheduler_info",
                json={"batch_size": 1, "mode": "decode"},
                timeout=check_connection_timeout,
            )
            update_scheduler_info_resp_status_code = (
                update_scheduler_info_resp.status_code
            )
            update_scheduler_info_resp_status_text = (
                update_scheduler_info_resp.json().get("status", "not ok")
            )
            update_scheduler_info_ok = (
                update_scheduler_info_resp_status_code == 200
                and update_scheduler_info_resp_status_text == "ok"
            )
            store_key_ok = _store_check_ok(
                tcp_store, f"{_SERVER_STARTUP_STORE_KEY_OK_PREFIX}{node_rank}"
            )
            return health_ok and update_scheduler_info_ok and store_key_ok
        except Exception as e:
            logging.warning(
                f"node rank {node_rank} health check failed, host={host}, port={port}, error={str(e)}"
            )
            return False

    while True:
        failed = _store_check_failed(tcp_store)
        if failed:
            logging.warning(f"other node server startup failed: {failed}")
            return False

        all_ready = True
        for node_rank, (_, ip, port) in enumerate(targets):
            if not _is_ready(node_rank, ip, port):
                all_ready = False
                break
        if all_ready:
            logging.info("all servers are ready")
            return True

        if time.time() - start_time > server_startup_timeout:
            logging.warning(f"waiting all servers startup timeout")
            return False

        time.sleep(check_interval)


def script_exit(pgrp_set: bool = False):
    sys.stdout.flush()
    if pgrp_set:
        os.killpg(0, signal.SIGKILL)
        os._exit(0)
    else:
        os._exit(0)


def try_upload_log(log_dir_path: str, upload_path: str):
    import shutil

    if not os.path.exists(log_dir_path) or not os.path.isdir(log_dir_path):
        logging.info(f"{log_dir_path} not exist, skip upload")
        return
    if not os.listdir(log_dir_path):
        logging.info(f"{log_dir_path} is empty, skip upload")
        return

    zip_path = f"{log_dir_path}.zip"
    shutil.make_archive(log_dir_path, "zip", log_dir_path)
    logging.info(f"zip {log_dir_path} to {zip_path}")

    logging.info(f"upload {zip_path} ...")
    os.system(f"osscmd put {zip_path} {upload_path}/{zip_path}")

    os.remove(zip_path)


if __name__ == "__main__":
    setup_logging()
    patch_logging_stream_handler()

    batch_size_list = json.loads(os.environ.get("BATCH_SIZE_LIST", "[1,4,8]"))
    input_len_list = json.loads(os.environ.get("INPUT_LEN_LIST", "[2048, 4096, 8192]"))
    is_decode = os.environ.get("IS_DECODE", "1") == "1"
    decode_test_length = int(os.environ.get("DECODE_TEST_LENGTH", 10))
    max_seq_len = max(input_len_list) + decode_test_length + 1

    os.environ["GEN_TIMELINE_SYNC"] = "1"
    os.environ["MAX_SEQ_LEN"] = str(max_seq_len)
    os.environ["FAKE_BALANCE_EXPERT"] = "1"
    os.environ.setdefault("WORKER_INFO_PORT_NUM", "10")
    os.environ["USE_BATCH_DECODE_SCHEDULER"] = "1"

    py_env_configs: PyEnvConfigs = setup_args()
    setup_and_configure_server(py_env_configs)

    port = py_env_configs.server_config.start_port
    world_rank = py_env_configs.parallelism_config.world_rank
    local_world_size = py_env_configs.parallelism_config.local_world_size
    log_dir_name = (
        f"test_output_{py_env_configs.model_args.model_type}_{py_env_configs.parallelism_config.dp_size}"
        f"_{py_env_configs.parallelism_config.tp_size}_{py_env_configs.parallelism_config.world_rank}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}"
    ).upper()
    log_dir_path = os.path.abspath(log_dir_name)
    os.makedirs(log_dir_path, exist_ok=True)

    os.environ["TORCH_CUDA_PROFILER_DIR"] = log_dir_path

    dist_config_str = os.environ.get(
        "GANG_CONFIG_STRING", py_env_configs.distribute_config.gang_config_string
    )
    if not dist_config_str:
        raise RuntimeError("no gang config string (GANG_CONFIG_STRING), unexpected!")
    node_world_size = len(members_from_test_env(dist_config_str))
    node_rank = world_rank // local_world_size

    tokenizer_path = py_env_configs.model_args.tokenizer_path
    if tokenizer_path is None:
        raise RuntimeError(
            f"fetch tokenizer path failed, tokenizer_path: {py_env_configs.model_args.tokenizer_path}"
        )

    input_query_dict = create_query(
        model_type=py_env_configs.model_args.model_type,
        tokenizer_path=tokenizer_path,
        input_len_list=input_len_list,
    )

    pgrp_set = False
    try:
        os.setpgrp()
        pgrp_set = True
    except Exception as e:
        logging.info(f"setpgrp error: {e}")

    request_tpot = 100
    bootstrap_timeout = 10
    server_startup_timeout = 1600
    retry_interval = 0.5
    retry_times = 3
    check_interval = 3
    check_connection_timeout = 10
    heartbeat_interval = 10

    os.environ["MAGA_SERVER_WORK_DIR"] = os.getcwd()
    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = log_dir_path
    server = LocalServerManager(port=port, log_dir=log_dir_path)

    tcp_store = _init_startup_store(
        py_env_configs, world_rank, timeout=bootstrap_timeout
    )
    if tcp_store is None:
        raise Exception("failed to init tcp store")

    ok_key = f"{_SERVER_STARTUP_STORE_KEY_OK_PREFIX}{node_rank}"
    logging.info(f"set startup store key {ok_key} to starting")
    _store_set_safe(tcp_store, ok_key, "starting")

    try:
        if not server.start_server(
            retry_interval=check_interval,
            check_connection_timeout=check_connection_timeout,
            timeout=server_startup_timeout,
        ):
            _store_set_safe(
                tcp_store,
                _SERVER_STARTUP_STORE_KEY_FAILED,
                json.dumps(
                    {
                        "world_rank": world_rank,
                        "host": socket.gethostname(),
                        "ip": socket.gethostbyname(socket.gethostname()),
                        "start_port": port,
                        "reason": "server.start_server() returned False (timeout or process exited)",
                    },
                    ensure_ascii=False,
                ),
            )
            server.print_process_log()
            raise Exception("server start failed")
        _store_set_safe(tcp_store, ok_key, "ok")

        if not wait_world_server_startup(
            tcp_store=tcp_store,
            py_env_configs=py_env_configs,
            check_interval=check_interval,
            check_connection_timeout=check_connection_timeout,
            server_startup_timeout=server_startup_timeout,
        ):
            raise Exception("wait world server startup failed")

        _store_barrier(tcp_store, node_rank=node_rank, node_world_size=node_world_size)

        if node_rank:
            logging.info(f"node rank non-zero: {node_rank}, wait for main.")
            wait_master_done(
                world_rank=world_rank,
                py_env_configs=py_env_configs,
                retry_interval=retry_interval,
                retry_times=retry_times,
                heartbeat_interval=heartbeat_interval,
                check_connection_timeout=check_connection_timeout,
            )

        else:
            logging.info(f"world rank zero: {world_rank}, start test")
            run_single(
                base_port=port,
                dp_size=py_env_configs.parallelism_config.dp_size,
                tp_size=py_env_configs.parallelism_config.tp_size,
                batch_size_list=batch_size_list,
                input_len_list=input_len_list,
                input_query_dict=input_query_dict,
                gang_config_string=py_env_configs.distribute_config.gang_config_string,
                local_world_size=py_env_configs.parallelism_config.local_world_size,
                request_tpot=request_tpot,
                connection_timeout=check_connection_timeout,
                retry_times=retry_times,
                retry_interval=retry_interval,
                is_decode=is_decode,
                dump_json_path=log_dir_path,
                decode_test_length=decode_test_length,
                is_speculative=False,
                propose_step=0,
                generate_config={},
            )

    finally:
        server.stop_server()
        upload_path = os.environ.get("UPLOAD_OSS_PATH", "")
        if upload_path != "":
            logging.info(f"upload log to {upload_path}")
            try_upload_log(log_dir_path, upload_path)
        script_exit(pgrp_set)
```

- [ ] **Step 2: Verify the file has no syntax errors**

Run:
```bash
python -c "import ast; ast.parse(open('rtp_llm/test/perf_test/multi_node/local_server_runner.py').read()); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 3: Verify all local imports resolve**

Run:
```bash
python -c "
from rtp_llm.test.perf_test.multi_node.perf_runner import run_single
from rtp_llm.test.perf_test.multi_node.perf_util import create_query
from rtp_llm.test.perf_test.multi_node.server_manager import LocalServerManager
from rtp_llm.test.perf_test.multi_node.perf_dataclass import ResponseInfo, MetricState
from rtp_llm.test.perf_test.multi_node.perf_impl import BatchPerfImpl
print('All local module imports OK')
"
```
Expected: `All local module imports OK`

- [ ] **Step 4: Commit**

```bash
git add rtp_llm/test/perf_test/multi_node/local_server_runner.py
git commit -m "refactor(multi_node): rewrite local_server_runner with self-contained local imports"
```

---

### Task 7: Verify no changes to external files

**Files:**
- None (verification only)

Verify that the original smoke test files are unchanged.

- [ ] **Step 1: Check git status for unintended changes**

Run:
```bash
git diff --name-only HEAD~6
```

Expected output should only contain files under `rtp_llm/test/perf_test/multi_node/` and `docs/superpowers/`:
```
docs/superpowers/specs/2026-04-23-multi-node-perf-benchmark-refactor-design.md
docs/superpowers/plans/2026-04-23-multi-node-perf-benchmark-refactor.md
rtp_llm/test/perf_test/multi_node/local_server_runner.py
rtp_llm/test/perf_test/multi_node/perf_dataclass.py
rtp_llm/test/perf_test/multi_node/perf_impl.py
rtp_llm/test/perf_test/multi_node/perf_runner.py
rtp_llm/test/perf_test/multi_node/perf_util.py
rtp_llm/test/perf_test/multi_node/server_manager.py
```

- [ ] **Step 2: Verify original files are untouched**

Run:
```bash
git diff rtp_llm/test/perf_test/batch_decode_test.py rtp_llm/test/perf_test/batch_perf_impl.py rtp_llm/test/perf_test/dataclass.py rtp_llm/test/perf_test/test_util.py rtp_llm/test/utils/maga_server_manager.py
```

Expected: No output (no changes)

---

### Task 8: End-to-end test on remote machine

**Files:**
- None (testing only)

Test on the ssh 211 remote machine in the container environment.

- [ ] **Step 1: SSH into the remote machine and enter the container**

Use the `setup-mi355x-env` skill to connect to the container environment on ssh 211.

- [ ] **Step 2: Navigate to the project directory and sync code**

Make sure the refactored code is available in the container. Either `git pull` or copy the files.

- [ ] **Step 3: Set environment variables**

```bash
export OMP_NUM_THREADS=8
export LD_PRELOAD=/opt/conda310/lib/libstdc++.so.6
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export HIP_VISIBLE_DEVICES=4,5,6,7
export SEQ_SIZE_PER_BLOCK=2048
export KERNEL_SEQ_SIZE_PER_BLOCK=16
export WARM_UP=0
export CONCURRENCY_LIMIT=128
export ENABLE_CUDA_GRAPH=0
export DECODE_CAPTURE_CONFIG=5,6,7,8
export QUANTIZATION=FP4_PER_GROUP_QUARK
export LOAD_PYTHON_MODEL=1
export LOAD_METHOD=fastsafetensors
export USE_ASM_PA=0
export WORLD_SIZE=4
export DP_SIZE=1
export TP_SIZE=4
export EP_SIZE=4
export DEVICE_RESERVE_MEMORY_BYTES=-16384000000
export RESERVER_RUNTIME_MEM_MB=40960
export MAX_SEQ_LEN=256000
export START_PORT=10666
export ACT_TYPE=bf16
export TOKENIZER_PATH=/mnt/nfs/RAID/shared/fangyuan/Qwen3.5-397B-A17B-MXFP4
export CHECKPOINT_PATH=/mnt/nfs/RAID/shared/fangyuan/Qwen3.5-397B-A17B-MXFP4
export MODEL_TYPE=qwen35_moe
export USE_ALL_GATHER=0
export USE_DEEPEP_MOE=0
export USE_DEEPEP_LOW_LATENCY=0
export USE_MORI_EP=1
export FT_SERVER_TEST=1
export ROCM_DISABLE_CUSTOM_AG=True
export FT_DISABLE_CUSTOM_AR=True
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=0
export RCCL_ENABLE_P2P=0
export DIST_COMM_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export HACK_LAYER_NUM=5
export FAKE_BALANCE_EXPERT=1
export GEN_TIMELINE_SYNC=1
export INPUT_LEN_LIST="[2048]"
export BATCH_SIZE_LIST="[128]"
export IS_DECODE=1
export DECODE_TEST_LENGTH=20
```

- [ ] **Step 4: Run the benchmark**

```bash
cd /path/to/rtp-llm
/opt/conda310/bin/python rtp_llm/test/perf_test/multi_node/local_server_runner.py
```

- [ ] **Step 5: Verify output artifacts**

After the benchmark completes, check the output directory:

```bash
# Find the output directory
ls -la TEST_OUTPUT_*

# Check for performance report JSON files
ls -la TEST_OUTPUT_*/*Result*.json

# Check for trace JSON files
ls -la TEST_OUTPUT_*/normal_*.json

# Check for process log
ls -la TEST_OUTPUT_*/main_logs/process.log
```

Expected:
- `Decode_Result.json` exists and contains valid JSON with metrics
- `normal_*.json` trace files exist (at least one)
- `main_logs/process.log` exists and contains server startup logs

- [ ] **Step 6: Verify PrettyTable output was printed**

Check the terminal output for a table like:
```
+----------+------------+------------------+--------------+------------------+-----------------+
|                                         Decode Result                                        |
+----------+------------+------------------+--------------+------------------+-----------------+
| Seq Len  | Batch Size | Sucess/Total Req | Input/Output | Waiting Time(ms) | Decode Time(ms) |
+----------+------------+------------------+--------------+------------------+-----------------+
| 2048     | 128        | 128/128          | ...          | ...              | ...             |
+----------+------------+------------------+--------------+------------------+-----------------+
```
