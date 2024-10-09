from typing import Dict, Any, List, NamedTuple
import prettytable as pt
import logging

def dump_model_to_table(config_map: Dict[str, Any]):
    return dump_config_to_table("MODEL CONFIG", config_map)

def dump_engine_to_table(config_map: Dict[str, Any]):
    return dump_config_to_table("ENGINE CONFIG", config_map)

def dump_config_to_table(title: str, config_map: Dict[str, Any]):
    table = pt.PrettyTable()
    table.title = title
    table.align = 'l'
    table.field_names = ["Options", "Values"]
    for option, value in config_map.items():
        table.add_row([option, value])    
    logging.info(table)

def dump_lora_infos_to_table(title: str, lora_infos: List[NamedTuple]):
    if len(lora_infos) == 0:
        logging.info("There is no lora_info")
        return
    
    table = pt.PrettyTable(lora_infos[0]._fields)
    table.title = title
    table.align = 'l'
    for lora_info in lora_infos:
        table.add_row(lora_info)
    logging.info(table)