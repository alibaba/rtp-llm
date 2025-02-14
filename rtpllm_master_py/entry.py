
import os
import time
import json
import logging
from typing import Dict, Any
from stub.librtpllm_master import PySubscribeConfigType, PySubscribeConfig, PyLoadbalanceConfig, PyEstimatorConfig, MasterInitParameter, RtpLLMMasterEntry

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    # subscribe config
    parser.add_argument('--cm_config_str', type=str, default="")
    parser.add_argument('--use_local', type=bool, default=False)
    parser.add_argument('--local_ip', type=str, default="127.0.0.1")
    parser.add_argument('--local_port', type=int, default=-1)
    # http server config
    parser.add_argument('--port', type=int, required=True)
    # model size config
    parser.add_argument("--model_size", type=float, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--force_replace_data_dir", type=str, default="")
    return parser.parse_args()


def find_config_dir_from_model_size(args: Any) -> Dict[str, str]:
    def _format_float(f: float):
        return int(f) if f.is_integer() else f

    formated_model_size = str(_format_float(args.model_size)) + "B"
    model_dir = f"{args.model_type}_{formated_model_size}"
    data_dir_path = args.force_replace_data_dir if args.force_replace_data_dir != "" else os.path.join(CUR_PATH, "data")
    data_path = os.path.join(data_dir_path, model_dir)
    if not os.path.exists(data_path):
        raise RuntimeError(f"Model size {formated_model_size} is not supported, data path {data_path} not exists")
    config_dict: Dict[str, str] = {}
    for file in os.listdir(data_path):
        if file.endswith(".json"):
            file_key = file.replace(".json", "")
            config_dict[file_key] = os.path.join(data_path, file)
    return config_dict


def create_estimator_config(args: Any):
    config_map = find_config_dir_from_model_size(args)
    estimator_config = PyEstimatorConfig()
    estimator_config.estimator_config_map = config_map
    return estimator_config


def create_load_balance_config(args: Any):
    load_balance_config = PyLoadbalanceConfig()
    if args.use_local == True:
        load_balance_config.subscribe_config.type = PySubscribeConfigType.LOCAL
        load_balance_config.subscribe_config.local_ip = args.local_ip
        load_balance_config.subscribe_config.local_http_port = args.local_port
        load_balance_config.subscribe_config.local_rpc_port = args.local_port + 1
    else:
        load_balance_config.subscribe_config.type = PySubscribeConfigType.CM2
        cm2_config_json = json.loads(args.cm_config_str)
        load_balance_config.subscribe_config.cluster_name = cm2_config_json['cm2_server_cluster_name']
        load_balance_config.subscribe_config.zk_host = cm2_config_json['cm2_server_zookeeper_host']
        load_balance_config.subscribe_config.zk_path = cm2_config_json['cm2_server_leader_path']
    return load_balance_config


if __name__ == '__main__':
    args = parse_args()
    master_init_param = MasterInitParameter()
    master_init_param.estimator_config = create_estimator_config(args)
    master_init_param.load_balance_config = create_load_balance_config(args)
    master_init_param.port = args.port
    entry = RtpLLMMasterEntry()
    if not entry.init(master_init_param):
        raise Exception("Failed to init master")
    logging.info("server start successfully")
    while True:
        time.sleep(100)