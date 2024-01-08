from typing import Dict, Any
import prettytable as pt

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
    print(table, flush=True)