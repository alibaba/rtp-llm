import logging
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from perf_dataclass import (
    MetricState,
    TableType,
    create_metrics_table,
)
from perf_impl import BatchPerfImpl


def run_single(
    base_port: int,
    dp_size: int,
    tp_size: int,
    batch_size_list: List[int],
    input_len_list: List[int],
    input_query_dict: Dict[int, str],
    is_decode: bool = True,
    dump_json_path: str = ".",
    decode_test_length: int = 10,
    is_speculative: bool = False,
    propose_step: int = 0,
    generate_config: Dict[str, Any] = {},
    gang_config_string: Optional[str] = None,
    local_world_size: int = 0,
    request_tpot: int = 100,
    connection_timeout: int = 10,
    retry_times: int = 3,
    retry_interval: float = 0.5,
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

    logging.info(f"start to run perf test")
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
