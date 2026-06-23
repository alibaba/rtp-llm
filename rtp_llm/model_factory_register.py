import logging
import os
import sys
import threading
from typing import Any, Dict, Iterable, List, Optional, Type

from rtp_llm.utils.import_util import (
    LazyModuleRegistry,
    import_optional_internal_source_entrypoint,
)

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

_model_factory: Dict[str, Type[Any]] = {}
_model_registry_lock = threading.RLock()
_lazy_model_registry = LazyModuleRegistry("model")
_model_type_to_module = _lazy_model_registry.name_to_module
_lazy_model_modules = _lazy_model_registry.module_paths
_loaded_model_modules = _lazy_model_registry.loaded_modules
_internal_lazy_models_registered = False
_internal_legacy_models_loaded = False


def register_model(
    name: str,
    model_type: Any,
    support_architectures: Optional[List[str]] = None,
    support_hf_repos: Optional[List[str]] = None,
):
    global _model_factory
    with _model_registry_lock:
        if name in _model_factory and _model_factory[name] != model_type:
            raise Exception(
                f"try register model {name} with type {_model_factory[name]} and {model_type}, confict!"
            )
        _model_factory[name] = model_type

    for architecture in support_architectures or []:
        register_hf_architecture(architecture, name)

    for repo in support_hf_repos or []:
        register_hf_repo(repo, name)


_hf_architecture_2_ft = {}


def register_hf_architecture(name: str, model_type: str):
    global _hf_architecture_2_ft
    with _model_registry_lock:
        if name in _hf_architecture_2_ft and _hf_architecture_2_ft[name] != model_type:
            raise Exception(
                f"try register model {name} with type {_hf_architecture_2_ft[name]} and {model_type}, confict!"
            )
        logging.debug("registerhf_architecture: %s -> %s", name, model_type)
        _hf_architecture_2_ft[name] = model_type


_hf_repo_2_ft = {}


def register_hf_repo(name: str, model_type: str):
    global _hf_repo_2_ft
    with _model_registry_lock:
        if name in _hf_repo_2_ft and _hf_repo_2_ft[name] != model_type:
            raise Exception(
                f"try register model {name} with type {_hf_repo_2_ft[name]} and {model_type}, confict!"
            )
        logging.debug("register_hf_repo: %s -> %s", name, model_type)
        _hf_repo_2_ft[name] = model_type


def register_lazy_model(
    name: str,
    module_path: str,
    support_architectures: Optional[Iterable[str]] = None,
    support_hf_repos: Optional[Iterable[str]] = None,
) -> None:
    """Register lightweight metadata for a model module.

    Importing the module is intentionally deferred until ModelFactory needs the
    concrete class. Architecture/repo mappings stay available for config.json
    inference without importing every model implementation.
    """
    _lazy_model_registry.register(name, module_path)

    for architecture in support_architectures or []:
        register_hf_architecture(architecture, name)
    for repo in support_hf_repos or []:
        register_hf_repo(repo, name)


def _import_model_module(module_path: str) -> None:
    _lazy_model_registry.import_module(module_path)


def get_lazy_model_module_path(model_type: Optional[str]) -> Optional[str]:
    if not model_type:
        return None
    _load_internal_lazy_models()
    return _lazy_model_registry.get_module_path(model_type)


def _load_internal_lazy_models() -> None:
    global _internal_lazy_models_registered
    with _model_registry_lock:
        if _internal_lazy_models_registered:
            return
        import_optional_internal_source_entrypoint("models.lazy_register")
        _internal_lazy_models_registered = True


def _load_internal_legacy_models() -> None:
    global _internal_legacy_models_loaded
    with _model_registry_lock:
        if _internal_legacy_models_loaded:
            return
        import_optional_internal_source_entrypoint("models.internal_init")
        _internal_legacy_models_loaded = True


def ensure_model_registered(model_type: str) -> bool:
    """Import the module that registers model_type, if it is known."""
    if not model_type:
        return False
    _load_internal_lazy_models()
    with _model_registry_lock:
        if model_type in _model_factory:
            return True
    module_path = _lazy_model_registry.get_module_path(model_type)
    if not module_path:
        _load_internal_legacy_models()
        with _model_registry_lock:
            if model_type in _model_factory:
                return True
        module_path = _lazy_model_registry.get_module_path(model_type)
        if not module_path:
            return False
    _import_model_module(module_path)
    with _model_registry_lock:
        return model_type in _model_factory


def ensure_all_models_registered() -> None:
    """Load every known model module.

    This is a compatibility escape hatch for tooling that still expects the old
    eager-registration side effect of ``import rtp_llm.models``.
    """
    _load_internal_lazy_models()
    _lazy_model_registry.import_all_modules()
    _load_internal_legacy_models()


def _register_builtin_lazy_models() -> None:
    register_lazy_model("bloom", "rtp_llm.models.bloom", ["BloomForCausalLM"])
    register_lazy_model(
        "chatglm2",
        "rtp_llm.models.chat_glm_v2",
        ["ChatGLMModel"],
        ["THUDM/chatglm2-6b", "THUDM/chatglm2-6b-int4", "THUDM/chatglm2-6b-32k"],
    )
    register_lazy_model("chat_glm_2", "rtp_llm.models.chat_glm_v2")
    register_lazy_model(
        "chatglm3",
        "rtp_llm.models.chat_glm_v3",
        support_hf_repos=[
            "THUDM/chatglm3-6b",
            "THUDM/chatglm3-6b-base",
            "THUDM/chatglm3-6b-32k",
        ],
    )
    register_lazy_model("chat_glm_3", "rtp_llm.models.chat_glm_v3")
    register_lazy_model(
        "chatglm4",
        "rtp_llm.models.chat_glm_v4",
        support_hf_repos=["THUDM/glm4-9b-chat", "THUDM/glm-4-9b-chat"],
    )
    register_lazy_model(
        "chatglm4v",
        "rtp_llm.models.chat_glm_v4_vision",
        support_hf_repos=["THUDM/glm-4v-9b"],
    )
    register_lazy_model(
        "cosyvoice_qwen", "rtp_llm.models.cosyvoice_qwen", ["CosyQwen2ForCausalLM"]
    )
    register_lazy_model(
        "deepseek2", "rtp_llm.models.deepseek_v2", ["DeepseekV2ForCausalLM"]
    )
    register_lazy_model(
        "deepseek3", "rtp_llm.models.deepseek_v2", ["DeepseekV3ForCausalLM"]
    )
    register_lazy_model(
        "deepseek-v3-mtp", "rtp_llm.models.deepseek_v2", ["DeepseekV3ForCausalLMNextN"]
    )
    register_lazy_model("kimi_k2", "rtp_llm.models.deepseek_v2")
    register_lazy_model("deepseek_v31", "rtp_llm.models.deepseek_v2")
    register_lazy_model(
        "deepseek_v32", "rtp_llm.models.deepseek_v2", ["DeepseekV32ForCausalLM"]
    )
    register_lazy_model("glm_5", "rtp_llm.models.deepseek_v2", ["GlmMoeDsaForCausalLM"])
    # REBASE CONFLICT CONTEXT(e2e00e570): source branch registered GLM5 MTP via
    # eager import side effects in `rtp_llm.models.__init__`. New base uses lazy
    # registration, so the MTP architecture must be listed explicitly here.
    register_lazy_model(
        "glm_5_mtp", "rtp_llm.models.deepseek_v2", ["GlmMoeDsaMtpForCausalLM"]
    )
    register_lazy_model(
        "deepseek_v4", "rtp_llm.models.deepseek_v4", ["DeepseekV4ForCausalLM"]
    )
    register_lazy_model(
        "deepseek_v4_mtp", "rtp_llm.models.deepseek_v4", ["DeepseekV4ForCausalLMNextN"]
    )
    register_lazy_model(
        "deepseek_vl_v2",
        "rtp_llm.models.deepseek_vl2.deepseek_vl2",
        ["DeepseekVL2ForCausalLM"],
    )
    register_lazy_model("falcon", "rtp_llm.models.falcon", ["FalconForCausalLM"])
    register_lazy_model("gpt_neox", "rtp_llm.models.gpt_neox", ["GPTNeoXForCausalLM"])
    register_lazy_model("gpt_neox_13b", "rtp_llm.models.gpt_neox")
    register_lazy_model(
        "llama", "rtp_llm.models.llama", ["LlamaForCausalLM", "YiForCausalLM"]
    )
    register_lazy_model("internlm", "rtp_llm.models.llama", ["InternLMForCausalLM"])
    register_lazy_model("internlm2", "rtp_llm.models.llama", ["InternLM2ForCausalLM"])
    register_lazy_model("xverse", "rtp_llm.models.llama", ["XverseForCausalLM"])
    register_lazy_model("aquila", "rtp_llm.models.llama", ["AquilaModel"])
    register_lazy_model("mistral", "rtp_llm.models.llama", ["MistralForCausalLM"])
    register_lazy_model("baichuan", "rtp_llm.models.llama", ["BaichuanForCausalLM"])
    register_lazy_model("baichuan2", "rtp_llm.models.llama")
    register_lazy_model("gemma", "rtp_llm.models.llama", ["GemmaForCausalLM"])
    register_lazy_model("cohere", "rtp_llm.models.llama", ["CohereForCausalLM"])
    register_lazy_model("mpt", "rtp_llm.models.mpt")
    register_lazy_model("phi", "rtp_llm.models.phi")
    register_lazy_model("qwen", "rtp_llm.models.qwen", ["QWenLMHeadModel"])
    register_lazy_model("qwen_7b", "rtp_llm.models.qwen")
    register_lazy_model("qwen_13b", "rtp_llm.models.qwen")
    register_lazy_model("qwen_1b8", "rtp_llm.models.qwen")
    register_lazy_model("qwen_2", "rtp_llm.models.qwen_v2", ["Qwen2ForCausalLM"])
    register_lazy_model("qwen_agent", "rtp_llm.models.qwen_v2")
    register_lazy_model("qwen_2_embedding", "rtp_llm.models.qwen_v2")
    register_lazy_model("qwen_tool", "rtp_llm.models.qwen_v2")
    register_lazy_model("qwen_2-mtp", "rtp_llm.models.qwen_v2")
    register_lazy_model(
        "qwen_2_moe", "rtp_llm.models.qwen_v2_moe", ["Qwen2MoeForCausalLM"]
    )
    register_lazy_model("qwen_3", "rtp_llm.models.qwen_v3", ["Qwen3ForCausalLM"])
    register_lazy_model("qwen_3_tool", "rtp_llm.models.qwen_v3")
    register_lazy_model(
        "qwen_3_moe", "rtp_llm.models.qwen_v3_moe", ["Qwen3MoeForCausalLM"]
    )
    register_lazy_model(
        "qwen_3_moe_eagle3", "rtp_llm.models.qwen_v3_moe", ["Qwen3MoeForCausalLMEagle"]
    )
    register_lazy_model("qwen3_coder_moe", "rtp_llm.models.qwen_v3_moe")
    register_lazy_model(
        "qwen3_next", "rtp_llm.models.qwen3_next.qwen3_next", ["Qwen3NextForCausalLM"]
    )
    register_lazy_model(
        "qwen35_moe",
        "rtp_llm.models.qwen3_next.qwen3_next",
        ["Qwen3_5MoeForConditionalGeneration"],
    )
    register_lazy_model(
        "qwen35_dense",
        "rtp_llm.models.qwen3_next.qwen3_next",
        ["Qwen3_5ForConditionalGeneration"],
    )
    register_lazy_model(
        "qwen3_next_mtp",
        "rtp_llm.models.qwen3_next.qwen3_next_mtp",
        ["Qwen3NextMTPForCausalLM"],
    )
    register_lazy_model(
        "qwen35_moe_mtp",
        "rtp_llm.models.qwen3_next.qwen3_next_mtp",
        ["Qwen35MoeMTPForCausalLM"],
    )
    register_lazy_model("qwen_vl", "rtp_llm.models.qwen_vl", ["QWenMLMHeadModel"])
    register_lazy_model(
        "qwen2_vl",
        "rtp_llm.models.qwen2_vl",
        ["Qwen2VLForConditionalGeneration"],
    )
    register_lazy_model(
        "qwen2_5_vl",
        "rtp_llm.models.qwen2_vl",
        ["Qwen2_5_VLForConditionalGeneration"],
    )
    register_lazy_model(
        "qwen3_vl",
        "rtp_llm.models.qwen3_vl",
        ["Qwen3VLForConditionalGeneration"],
    )
    register_lazy_model(
        "qwen3_vl_moe",
        "rtp_llm.models.qwen3_vl_moe",
        ["Qwen3VLMoeForConditionalGeneration"],
    )
    register_lazy_model("qwen_v2_audio", "rtp_llm.models.qwen_v2_audio.qwen_v2_audio")
    register_lazy_model("internvl", "rtp_llm.models.internvl", ["InternVLChatModel"])
    register_lazy_model("llava", "rtp_llm.models.llava", ["LlavaLlamaForCausalLM"])
    register_lazy_model("minicpmv", "rtp_llm.models.minicpmv.minicpmv", ["MiniCPMV"])
    register_lazy_model(
        "minicpmv_embedding",
        "rtp_llm.models.minicpmv_embedding.minicpmv_embedding",
        ["MiniCPMVEmbedding"],
    )
    register_lazy_model(
        "bert",
        "rtp_llm.models.bert",
        ["BertModel", "BertForMaskedLM", "BertForSequenceClassification"],
    )
    register_lazy_model(
        "roberta",
        "rtp_llm.models.bert",
        ["XLMRobertaModel", "RobertaModel", "XLMRobertaForSequenceClassification"],
    )
    register_lazy_model(
        "jina_bert_code",
        "rtp_llm.models.jina_bert.jina_bert",
        support_hf_repos=["jinaai/jina-bert-v2-qk-post-norm"],
    )
    register_lazy_model(
        "megatron_bert", "rtp_llm.models.megatron_bert", ["MegatronBertModel"]
    )
    register_lazy_model("mixtral", "rtp_llm.models.mixtral", ["MixtralForCausalLM"])
    register_lazy_model(
        "glm4_moe", "rtp_llm.models.glm4_moe", support_hf_repos=["Glm4MoeForCausalLM"]
    )
    register_lazy_model(
        "gpt_bigcode", "rtp_llm.models.starcoder", ["GPTBigCodeForCausalLM"]
    )
    register_lazy_model("wizardcoder", "rtp_llm.models.starcoder")
    register_lazy_model(
        "starcoder2", "rtp_llm.models.starcoder2", ["Starcoder2ForCausalLM"]
    )


_register_builtin_lazy_models()
_load_internal_lazy_models()


class ModelDict:
    @staticmethod
    def get_ft_model_type_by_hf_repo(repo: str) -> Optional[str]:
        global _hf_repo_2_ft
        model_type = _hf_repo_2_ft.get(repo, None)
        if model_type is None:
            _load_internal_legacy_models()
            model_type = _hf_repo_2_ft.get(repo, None)
        logging.debug("get hf_repo model type: %s, %s", repo, model_type)
        return model_type

    @staticmethod
    def get_ft_model_type_by_hf_architectures(architecture):
        global _hf_architecture_2_ft
        model_type = _hf_architecture_2_ft.get(architecture, None)
        if model_type is None:
            _load_internal_legacy_models()
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
