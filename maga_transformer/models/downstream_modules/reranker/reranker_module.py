from typing import Dict, List, Any, Tuple
from transformers import PreTrainedTokenizerBase

from maga_transformer.models.downstream_modules.custom_module import CustomModule, CustomRenderer
from maga_transformer.models.downstream_modules.common_input_generator import CommonInputGenerator
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

from maga_transformer.async_decoder_engine.embedding.embedding_stream import EngineInputs, EngineOutputs
from maga_transformer.models.downstream_modules.reranker.api_datatype import VoyageRerankerRequest, VoyageRerankerResponse, RankingItem
from maga_transformer.models.downstream_modules.classifier.classifier import ClassifierHandler
from maga_transformer.models.downstream_modules.classifier.roberta_classifier import RobertaClassifierHandler


class RerankerModule(CustomModule):
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.renderer = RerankerRenderer(self.config_, self.tokenizer_)
        self.handler = ClassifierHandler(self.config_)
        
class RobertaRerankerModule(CustomModule):
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.renderer = RerankerRenderer(self.config_, self.tokenizer_)
        self.handler = RobertaClassifierHandler(self.config_)

class RerankerRenderer(CustomRenderer):
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.generator = CommonInputGenerator(tokenizer, config)

    async def render_request(self, request: Dict[str, Any]):
        return VoyageRerankerRequest(**request)

    async def create_input(self, formated_request: VoyageRerankerRequest):
        input: List[Tuple[str, str]] = [(formated_request.query, doc) for doc in formated_request.documents]
        return await self.generator.generate(input, truncate=formated_request.truncation)

    async def render_response(self, formated_request: VoyageRerankerRequest, inputs: EngineInputs, outputs: EngineOutputs) -> Dict[str, Any]:
        rank_items = [RankingItem(index=i, document=formated_request.documents[i], relevance_score=float(outputs.outputs[i])) for i in range(len(formated_request.documents))]
        rank_items.sort(key=lambda x: x.relevance_score, reverse=True)
        if formated_request.top_k is not None:
            rank_items = rank_items[: min(len(rank_items), formated_request.top_k)]
        return VoyageRerankerResponse(results=rank_items, total_tokens=len(inputs.token_ids)).model_dump()