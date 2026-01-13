import logging
from typing import Any, Dict, List, Optional, Type

_model_factory: Dict[str, Type[Any]] = {}


def register_model(
    name: str,
    model_type: Any,
    support_architectures: List[str] = [],
    support_hf_repos: List[str] = [],
):
    global _model_factory
    if name in _model_factory and _model_factory[name] != model_type:
        raise Exception(
            f"try register model {name} with type {_model_factory[name]} and {model_type}, confict!"
        )
    _model_factory[name] = model_type

    for architecture in support_architectures:
        register_hf_architecture(architecture, name)

    for repo in support_hf_repos:
        register_hf_repo(repo, name)


_hf_architecture_2_ft = {}


def register_hf_architecture(name: str, model_type: str):
    global _hf_architecture_2_ft
    if name in _hf_architecture_2_ft and _hf_architecture_2_ft[name] != model_type:
        raise Exception(
            f"try register model {name} with type {_hf_architecture_2_ft[name]} and {model_type}, confict!"
        )
    logging.debug("registerhf_architecture: %s -> %s", name, model_type)
    _hf_architecture_2_ft[name] = model_type


_hf_repo_2_ft = {}


def register_hf_repo(name: str, model_type: str):
    global _hf_repo_2_ft
    if name in _hf_repo_2_ft and _hf_repo_2_ft[name] != model_type:
        raise Exception(
            f"try register model {name} with type {_hf_repo_2_ft[name]} and {model_type}, confict!"
        )
    logging.debug("register_hf_repo: %s -> %s", name, model_type)
    _hf_repo_2_ft[name] = model_type


class ModelDict:
    @staticmethod
    def get_ft_model_type_by_hf_repo(repo: str) -> Optional[str]:
        global _hf_repo_2_ft
        model_type = _hf_repo_2_ft.get(repo, None)
        logging.debug("get hf_repo model type: %s, %s", repo, model_type)
        return model_type

    @staticmethod
    def get_ft_model_type_by_hf_architectures(architecture):
        global _hf_architecture_2_ft
        model_type = _hf_architecture_2_ft.get(architecture, None)
        logging.debug("get architectur model type: %s, %s", architecture, model_type)
        return model_type

    @staticmethod
    def get_ft_model_type_by_config(config: Dict[str, Any]) -> Optional[str]:
        if config.get("architectures", []):
            # hack for ChatGLMModel: chatglm and chatglm2 use same architecture
            architecture = config.get("architectures")[0]
            if architecture in ["ChatGLMModel", "ChatGLMForConditionalGeneration"]:
                _name_or_path = config.get("_name_or_path", "")
                if (
                    not config.get("multi_query_attention", False)
                    or "chatglm-6b" in _name_or_path
                ):
                    return "chatglm"
                elif "chatglm3" in _name_or_path:
                    return "chatglm3"
                elif "glm-4-" in _name_or_path:
                    return "chatglm4"
                elif "glm-4v" in _name_or_path:
                    return "chatglm4v"
                else:
                    return "chatglm2"
            if architecture == "QWenLMHeadModel":
                if config.get("visual"):
                    if config["visual"].get("layers"):
                        return "qwen_vl"
                    else:
                        return "qwen_vl_1b8"
            if architecture == "BaichuanForCausalLM":
                vocab_size = config.get("vocab_size", 64000)
                if vocab_size == 125696:
                    return "baichuan2"
                else:
                    return "baichuan"
            if architecture == "GPTNeoXForCausalLM":
                vocab_size = config.get("vocab_size", 50432)
                if vocab_size == 250752:
                    return "gpt_neox_13b"
                else:
                    return "gpt_neox"
            return ModelDict.get_ft_model_type_by_hf_architectures(architecture)
        else:
            logging.warning(f"config have no architectures: {config}")
        return None
