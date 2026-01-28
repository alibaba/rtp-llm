import logging
import os
import random
import time
from typing import Any, Dict, List

import prettytable as pt
from odps import ODPS
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rtp_llm.utils.fuser import fetch_remote_file_to_local


from pytest import mark
def write_odps(table_name: str, records: List[Any], fields: List[str] = []):
    table = pt.PrettyTable(align="l")
    table.field_names = fields
    table.add_rows(records)
    if "framework" in table.field_names:
        table.del_column("framework")
    if "commit" in table.field_names:
        table.del_column("commit")
    logging.info(f"records: \n{table.get_string()}")

    if "ODPS_PROJECT" not in os.environ:
        logging.warning("no odps config")
        return
    partition_name = os.environ["PARTITION_NAME"]
    odps = ODPS(
        os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID"),
        os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
        os.environ["ODPS_PROJECT"],
        endpoint="http://service-corp.odps.aliyun-inc.com/api",
    )

    table = odps.get_table(table_name)
    if table.exist_partition(partition_name):
        _ = table.get_partition(partition_name)
    else:
        _ = table.create_partition(partition_name)

    retry_limit = 10
    retry = 1
    while True:
        try:
            with table.open_writer(partition=partition_name) as writer:
                writer.write(records)
            break
        except Exception as e:
            logging.warning("%s", str(e))
            time.sleep(random.random() * 10 * retry)
            retry += 1
            if retry > retry_limit:
                raise e


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

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return {x: _create_query_single(tokenizer, x) for x in input_len_list}
