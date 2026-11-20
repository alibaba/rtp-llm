from rtp_llm.model_factory_register import register_model
from rtp_llm.models.bert import Bert, BertWeightInfo
from rtp_llm.models.vision_bert.configuration_mbert import MBertRelConfig
from rtp_llm.utils.util import get_config_from_path

SEP_START_TOKEN_ID = -1
SEP_END_TOKEN_ID = -2
SEP_PADDING_ID = -3


class VisionBert(Bert):
    def support_cuda_graph(self) -> bool:
        # Vision features and their token locations are request-owned tensors;
        # the shared graph runner does not capture or replay them.
        return False

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = Bert._create_config(ckpt_path)
        config.mm_model_config.include_sep_tokens = True
        config.mm_model_config.mm_sep_tokens = [[SEP_START_TOKEN_ID, SEP_END_TOKEN_ID]]
        config.mm_related_params.preprocess_batch_size = -1  # all in one batch

        config_json = get_config_from_path(config.ckpt_path)
        mbert_rel_config: MBertRelConfig = MBertRelConfig.from_dict(config_json)
        config.mm_related_params.config["vision_seq_len"] = (
            mbert_rel_config.vision_seq_len
        )
        config.mm_related_params.config["vision_hidden_size"] = (
            mbert_rel_config.vision_hidden_size
        )
        config.mm_related_params.config["ckpt_path"] = config.ckpt_path
        config.mm_model_config.is_multimodal = True
        return config

    @staticmethod
    def get_weight_cls():
        return BertWeightInfo


register_model("vision_bert", VisionBert, ["VisionBert"])
