import threading
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs
from rtp_llm.config.exceptions import EmbeddingInputException, ExceptionType
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.models.downstream_modules.reranker.api_datatype import (
    VoyageRerankerRequest,
)
from rtp_llm.models.downstream_modules.reranker.reranker_module import RerankerRenderer
from rtp_llm.utils.model_weight import CkptWeightInfo
from rtp_llm.utils.tensor_utils import get_last_token_from_combo_tokens
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import to_torch_dtype


class Qwen3RerankerModule(CustomModule):

    def __init__(self, config: ModelConfig, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.renderer = Qwen3RerankerRenderer(self.config_, self.tokenizer_)
        token_false_id = self.tokenizer_.convert_tokens_to_ids("no")
        token_true_id = self.tokenizer_.convert_tokens_to_ids("yes")
        self.handler = Qwen3RerankerHandler(self.config_, token_false_id, token_true_id)


class Qwen3RerankerRenderer(RerankerRenderer):
    def __init__(self, config: ModelConfig, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens_ = self.tokenizer_.encode(
            prefix, add_special_tokens=False, return_tensors="np"
        )[0]
        self.suffix_tokens_ = self.tokenizer_.encode(
            suffix, add_special_tokens=False, return_tensors="np"
        )[0]
        self.lock = threading.Lock()

    def format_instruction(
        self, instruction: Optional[str], query: str, doc: str
    ) -> str:
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = (
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )
        )
        return output

    def create_input(self, formated_request: VoyageRerankerRequest):
        if self.config_.qwen3_reranker_input_format == "raw":
            return self._create_raw_input(formated_request)
        with self.lock:
            begin_time = current_time_ms()
            input: List[str] = [
                self.format_instruction(
                    formated_request.instruction, formated_request.query, doc
                )
                for doc in formated_request.documents
            ]
            inputs = self.tokenizer_(
                input,
                padding=False,
                truncation="longest_first",
                return_tensors="np",
                return_attention_mask=False,
                max_length=self.config_.max_seq_len
                - len(self.prefix_tokens_)
                - len(self.suffix_tokens_),
            )
            inputs = [
                np.concatenate([self.prefix_tokens_, input, self.suffix_tokens_])
                for input in inputs["input_ids"]
            ]
            input_lengths_t = torch.tensor([len(input) for input in inputs]).to(
                torch.int32
            )
            input_t = torch.from_numpy(np.concatenate(inputs)).to(torch.int32)
            kmonitor.report(
                GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time
            )
            kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(input_t))

            return EngineInputs(
                token_ids=input_t,
                token_type_ids=torch.zeros_like(input_t),
                input_lengths=input_lengths_t,
                multimodal_inputs=[],
            )

    def _create_raw_input(self, formated_request: VoyageRerankerRequest):
        """Concatenate query and document tokens without the reranker prompt template.

        This is the token stream vLLM scores for score(query, doc) when a Qwen3
        reranker is served with is_original_qwen3_reranker and no chat template.
        Query and document are encoded separately, so the tokens at the boundary
        can differ from encoding the concatenated string.

        Length handling (max_seq_len budget per pair):
        - truncation=False: a pair over the budget is rejected, like vLLM without
          truncate_prompt_tokens.
        - truncation=True: the pair is cut with the "longest_first" pair strategy
          of HF fast tokenizers (tokens are removed from the end of the longer
          side first), so a long query cannot drop the document entirely.
        Requests with no documents or with an empty (zero-token) pair are rejected.
        """
        if not formated_request.documents:
            raise EmbeddingInputException(
                ExceptionType.EMPTY_PROMPT_ERROR, "documents should not be empty"
            )
        max_len = self.config_.max_seq_len
        with self.lock:
            begin_time = current_time_ms()
            query_tokens = self.tokenizer_.encode(
                formated_request.query, add_special_tokens=False
            )
            inputs = []
            for index, doc in enumerate(formated_request.documents):
                doc_tokens = self.tokenizer_.encode(doc, add_special_tokens=False)
                total_len = len(query_tokens) + len(doc_tokens)
                if total_len == 0:
                    raise EmbeddingInputException(
                        ExceptionType.EMPTY_PROMPT_ERROR,
                        f"query and documents[{index}] are both empty",
                    )
                if total_len > max_len:
                    if not formated_request.truncation:
                        raise EmbeddingInputException(
                            ExceptionType.LONG_PROMPT_ERROR,
                            f"query + documents[{index}] has {total_len} tokens, "
                            f"exceeds max_seq_len {max_len} with truncation disabled",
                        )
                    query_len, doc_len = _longest_first_lengths(
                        len(query_tokens), len(doc_tokens), max_len
                    )
                    tokens = query_tokens[:query_len] + doc_tokens[:doc_len]
                else:
                    tokens = query_tokens + doc_tokens
                inputs.append(np.array(tokens, dtype=np.int32))
            input_lengths_t = torch.tensor([len(input) for input in inputs]).to(
                torch.int32
            )
            input_t = torch.from_numpy(np.concatenate(inputs)).to(torch.int32)
            kmonitor.report(
                GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time
            )
            kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(input_t))

            return EngineInputs(
                token_ids=input_t,
                token_type_ids=torch.zeros_like(input_t),
                input_lengths=input_lengths_t,
                multimodal_inputs=[],
            )


def _longest_first_lengths(query_len: int, doc_len: int, max_len: int):
    """Pair lengths after "longest_first" truncation to max_len tokens.

    Mirrors the HF fast (Rust) tokenizers implementation used by Qwen tokenizers
    and vLLM: the shorter side is kept when it fits in half of the budget,
    otherwise both sides get half and the originally longer side gets the odd
    token, with the document winning ties.
    """
    if query_len + doc_len <= max_len:
        return query_len, doc_len
    short_len, long_len = sorted((query_len, doc_len))
    if short_len > max_len:
        long_len = short_len
    else:
        long_len = max(short_len, max_len - short_len)
    if short_len + long_len > max_len:
        short_len = max_len // 2
        long_len = short_len + max_len % 2
    if query_len > doc_len:
        return long_len, short_len
    return short_len, long_len


class Qwen3RerankerHandler(CustomHandler):

    def __init__(self, config: ModelConfig, token_false_id: int, token_true_id: int):
        super().__init__(config)
        self.token_false_id = token_false_id
        self.token_true_id = token_true_id
        self.tie_word_embeddings = config.tie_word_embeddings
        self.lm_head_weight_name = (
            "model.embed_tokens.weight"
            if self.tie_word_embeddings
            else "lm_head.weight"
        )
        # Score with only the [no, yes] lm_head rows, without bias, in fp32.
        self.fixed_head: bool = config.qwen3_reranker_fixed_head
        # Set by init(): yes_no_weight ([2, hidden], fp32, rows [no, yes]) for the
        # fixed head, linear for the legacy full-vocab head.
        self.yes_no_weight: Optional[torch.Tensor] = None
        self.linear: Optional[torch.nn.Linear] = None

    def custom_weight_info(self) -> List[CustomAtomicWeight]:
        w_list = [self.lm_head_weight_name]
        weights = []
        for k in w_list:
            weights.append(
                CustomAtomicWeight(CustomAtomicWeight.prefix + k, [CkptWeightInfo(k)])
            )
        return weights

    def init(self, tensor_map: Dict[str, torch.Tensor]):
        data_type = to_torch_dtype(self.config_.data_type)
        linear_weight = tensor_map[self.lm_head_weight_name]
        if self.fixed_head:
            row_ids = torch.tensor(
                [self.token_false_id, self.token_true_id], device=linear_weight.device
            )
            self.yes_no_weight = (
                linear_weight.index_select(0, row_ids)
                .float()
                .to(self.device)
                .contiguous()
            )
            return
        self.linear = torch.nn.Linear(linear_weight.shape[1], linear_weight.shape[0])
        self.linear.weight.data = linear_weight
        self.linear = self.linear.to(data_type).eval().to(self.device)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        last_token = get_last_token_from_combo_tokens(hidden_states, input_lengths)
        if self.fixed_head:
            # [batch, 2] logits ordered as [no, yes]
            logits = last_token.float() @ self.yes_no_weight.t()
            return torch.sigmoid(logits[:, 1] - logits[:, 0])
        last_token_logits = self.linear(last_token)
        true_vector = last_token_logits[:, self.token_true_id]
        false_vector = last_token_logits[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp()
        return scores
