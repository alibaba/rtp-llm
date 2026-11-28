"""Qwen3 draft sampler; unsharded weight vocabulary is independent of LM-head TP."""

from rtp_llm.models_py.new_models.deepseek_v4_dspark.markov_head import DSparkMarkovHead
from rtp_llm.models_py.weight_mapper import WeightsMapper


class Qwen3DSparkMarkovHead(DSparkMarkovHead):
    def __init__(self, target_vocab, draft_vocab, rank):
        super().__init__(target_vocab, draft_vocab, rank)
        self.checkpoint_prefix = "markov_head."
        self.WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={self.checkpoint_prefix: ""})
