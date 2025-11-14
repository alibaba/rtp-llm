import logging
from typing import Any, Dict, List, NamedTuple

import prettytable as pt

def dump_lora_infos_to_table(title: str, lora_infos: List[NamedTuple]):
    if len(lora_infos) == 0:
        logging.info("There is no lora_info")
        return

    table = pt.PrettyTable(lora_infos[0]._fields)
    table.title = title
    table.align = "l"
    for lora_info in lora_infos:
        table.add_row(lora_info)
    logging.info(table)
