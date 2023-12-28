import logging

_hf_architecture_2_ft = {}
def register_hf_architecture(name: str, model_type: str):
    global _hf_architecture_2_ft
    if name in _hf_architecture_2_ft and _hf_architecture_2_ft[name] != model_type:
        raise Exception(f"try register model {name} with type {_hf_architecture_2_ft[name]} and {model_type}, confict!")
    _hf_architecture_2_ft[name] = model_type

_hf_repo_2_ft = {}

def register_hf_repo(name: str, model_type: str):
    global _hf_repo_2_ft
    if name in _hf_repo_2_ft and _hf_repo_2_ft[name] != model_type:
        raise Exception(f"try register model {name} with type {_hf_repo_2_ft[name]} and {model_type}, confict!")
    _hf_repo_2_ft[name] = model_type


class ModelDict:
    @staticmethod
    def get_ft_model_type_by_hf_repo(repo):
        global _hf_repo_2_ft
        return _hf_repo_2_ft.get(repo, None)

    @staticmethod
    def get_ft_model_type_by_hf_architectures(architecture):
        global _hf_architecture_2_ft
        return _hf_architecture_2_ft.get(architecture, None)

    @staticmethod
    def get_ft_model_type_by_config(config):
        if config.get('architectures', []):
            # hack for ChatGLMModel: chatglm and chatglm2 use same architecture
            architecture = config.get('architectures')[0]
            if architecture == 'ChatGLMModel':
                if not config.get('multi_query_attention', False) or 'chatglm-6b' in config.get('_name_or_path', ''):
                    return 'chatglm'
                elif 'chatglm3' in config.get('_name_or_path', ''):
                    return 'chatglm3'
                else:
                    return 'chatglm2'
            if architecture == 'QWenLMHeadModel':
                if config.get('visual'):
                    return 'qwen_vl'
            logging.info(f"get architecture: {architecture} model_type")
            return ModelDict.get_ft_model_type_by_hf_architectures(architecture)   
        return None  
    
register_hf_architecture("GPTNeoXForCausalLM", "gpt_neox")
register_hf_architecture("BaichuanForCausalLM", "baichuan")
register_hf_architecture("BaiChuanForCausalLM", "baichuan")
register_hf_architecture("BloomForCausalLM", "bloom")
register_hf_architecture("ChatGLMForConditionalGeneration", "")
register_hf_architecture("ChatGLMModel", "chatglm2")
# register_hf_architecture("CLIPModel", "")
# register_hf_architecture("GPTBigCodeForCausalLM", "")
# register_hf_architecture("InternLMForCausalLM", "")
register_hf_architecture("LlamaForCausalLM", "llama")
# register_hf_architecture("LlavaLlamaForCausalLM", "")
register_hf_architecture("MixFormerSequentialForCausalLM", "")
# register_hf_architecture("MPTForCausalLM", "")
register_hf_architecture("QWenLMHeadModel", "qwen_7b")
# register_hf_architecture("RWForCausalLM", "")
register_hf_architecture("YiForCausalLM", "llama")
register_hf_architecture("FalconForCausalLM", "falcon")
register_hf_architecture("LlavaLlamaForCausalLM", "llava")
register_hf_architecture("LlavaTuringForCausalLM", "turing_005_vl")

# fix chatglm architectures一样，但是ft 类型不一样
register_hf_repo("THUDM/chatglm-6b", "chatglm")
register_hf_repo("THUDM/chatglm-6b-int4", "chatglm")
register_hf_repo("THUDM/chatglm-6b-int4-qe", "chatglm")
register_hf_repo("THUDM/chatglm-6b-int8", "chatglm")
register_hf_repo("THUDM/chatglm2-6b", "chatglm2")
register_hf_repo("THUDM/chatglm2-6b-int4", "chatglm2")
register_hf_repo("THUDM/chatglm2-6b-32k", "chatglm2")
register_hf_repo("THUDM/chatglm3-6b", "chatglm3")
register_hf_repo("THUDM/chatglm3-6b-base", "chatglm3")
register_hf_repo("THUDM/chatglm3-6b-32k", "chatglm3")

