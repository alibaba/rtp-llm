import json
import os


def load_num_labels(ckpt_path: str) -> int:
    config_path = os.path.join(ckpt_path, "config.json")
    with open(config_path, "r") as reader:
        config_json = json.loads(reader.read())
    if "num_labels" in config_json:
        return config_json["num_labels"]
    if "id2label" in config_json:
        return len(config_json["id2label"])
    raise Exception(
        "unknown label num, please set 'num_labels' or 'id2label' in config.json"
    )
