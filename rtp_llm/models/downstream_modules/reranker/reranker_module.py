from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.render.reranker_renderer import RerankerRenderer
from rtp_llm.models.downstream_modules.classifier.bert_classifier import (
    BertClassifierHandler,
)
from rtp_llm.models.downstream_modules.classifier.classifier import ClassifierHandler
from rtp_llm.models.downstream_modules.classifier.roberta_classifier import (
    RobertaClassifierHandler,
)
from rtp_llm.models.downstream_modules.custom_module import CustomModule
from rtp_llm.tokenizer_factory.tokenizers import BaseTokenizer


class RerankerModule(CustomModule):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.renderer = RerankerRenderer(self.config_, self.tokenizer_)
        self.handler = ClassifierHandler(self.config_)


class BertRerankerModule(CustomModule):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.renderer = RerankerRenderer(self.config_, self.tokenizer_)
        self.handler = BertClassifierHandler(self.config_)


class RobertaRerankerModule(CustomModule):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.renderer = RerankerRenderer(self.config_, self.tokenizer_)
        self.handler = RobertaClassifierHandler(self.config_)
