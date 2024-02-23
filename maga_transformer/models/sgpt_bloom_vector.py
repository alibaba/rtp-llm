
import torch
import numpy as np
from typing import Any, Dict, List

from maga_transformer.models.sgpt_bloom import SGPTBloom
from maga_transformer.models.base_model import GenerateOutput
from maga_transformer.model_factory_register import register_model

class SGPTBloomVector(SGPTBloom):
    @torch.no_grad()
    def generate_weighted_hidden_states_stream(self, input_token_ids: torch.IntTensor):  
        eos_token_id = self.config.special_tokens.eos_token_id
        batch_size = input_token_ids.shape[0]
        input_mask = torch.where(input_token_ids != eos_token_id, 1, 0)
        
        gen_output = list(self.generate_hidden_states_stream(input_token_ids))[0]
        hidden_states = gen_output.hidden_states

        weights = (
            torch.arange(start=1, end=hidden_states.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(hidden_states.size())
            .float().to(hidden_states.device)
        )

        # input_mask_expanded.shape = [bs, seq_len, hid_dim]
        # input_mask.shape = [batch, len]
        # input_mask_expanded.shape = [batch, len, feat]
        input_mask_expanded = (
            input_mask
            .unsqueeze(-1)
            .expand(hidden_states.size())
            .float()
        ).to(hidden_states.device)

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        embeddings = embeddings.cpu()

        norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        yield GenerateOutput(norm,
                            input_token_ids.unsqueeze(1), # add beam dim
                            torch.ones(batch_size),
                            [{"decimals": 6}] * batch_size)

    @staticmethod
    def process_encode_plugin(prompt: str, generate_config: Dict[str, Any], tokenizer: Any, max_seq_len: int, **kwargs: Any) -> List[int]:
        custon_gen_cfg = generate_config["custom_prop"]
        is_query = custon_gen_cfg.get("is_query", False)
        case_sensitive = custon_gen_cfg.get("case_sensitive", False)

        prompt = prompt if case_sensitive else prompt.lower() 
        tokenizer = tokenizer.tokenizer # PreTrainedTokenizerFast
        batch_tokens = tokenizer(prompt, padding=False, truncation=True, max_length=max_seq_len - 2)

        input_ids = batch_tokens["input_ids"]
        if is_query:
            SPECB_QUE_BOS = tokenizer.encode("[", add_special_tokens=False)[0]
            SPECB_QUE_EOS = tokenizer.encode("]", add_special_tokens=False)[0]
            input_ids.insert(0, SPECB_QUE_BOS)
            input_ids.append(SPECB_QUE_EOS)
        else:
            SPECB_DOC_BOS = tokenizer.encode("{", add_special_tokens=False)[0]
            SPECB_DOC_EOS = tokenizer.encode("}", add_special_tokens=False)[0]
            input_ids.insert(0, SPECB_DOC_BOS)
            input_ids.append(SPECB_DOC_EOS)

        return input_ids

    @torch.no_grad()
    def generate_stream(self,
                        input_token_ids, input_lengths, generate_config):
        return self.generate_weighted_hidden_states_stream(input_token_ids=input_token_ids)
    
register_model('sgpt_bloom_vector', SGPTBloomVector)
