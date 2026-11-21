"""V4.1 checkpoint names for the existing full-vocabulary DSpARK sampler."""

from rtp_llm.models_py.new_models.deepseek_v4_dspark.markov_head import DSparkMarkovHead
from rtp_llm.models_py.weight_mapper import WeightsMapper


class V41DSparkMarkovHead(DSparkMarkovHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoint_names = {
            self.checkpoint_prefix + "embed.weight": "markov_w1.weight",
            self.checkpoint_prefix + "head.weight": "markov_w2.weight",
        }
        self.WEIGHTS_MAPPER = WeightsMapper(exact_mapping=self._checkpoint_names)

    def checkpoint_weight_name_filter(self):
        return self._checkpoint_names.__contains__
