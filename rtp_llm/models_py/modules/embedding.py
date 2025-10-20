import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group, all_gather
from rtp_llm.ops.compute_ops import rtp_llm_ops


class EmbeddingTorch(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.weight)


class Embedding(nn.Module):
    def __init__(self, config: GptInitModelParameters, weight: torch.Tensor):
        super().__init__()
        self.weight = weight
        self.config = config

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tokens = input.size(0)
        hidden_size = self.weight.size(-1)
        output = torch.empty(
            (tokens, hidden_size), dtype=self.weight.dtype, device=input.device
        )
        rtp_llm_ops.embedding(output, input, self.weight.data)
        if self.config.tp_size > 1:
            m, n = output.shape
            output = all_gather(output, group=Group.TP)
            output = (
                output.reshape(self.config.tp_size, m, n)
                .transpose(0, 1)
                .contiguous()
                .reshape(m, -1)
            )
        return output


class EmbeddingBert(nn.Module):
    def __init__(self, config: GptInitModelParameters, weight: torch.Tensor):
        super().__init__()
        self.weight = weight
        self.config = config

    def forward(
        self,
        input: torch.Tensor,
        combo_position_ids: torch.Tensor,
        position_encoding: torch.Tensor,
        combo_tokens_type_ids: torch.Tensor,
        token_type_embedding: torch.Tensor,
        input_embedding_scalar: float,
    ) -> torch.Tensor:
        tokens = input.size(0)
        hidden_size = self.weight.size(-1)
        output = torch.empty(
            (tokens, hidden_size), dtype=self.weight.dtype, device=input.device
        )
        print("embedding bert start")
        print(f"output shape: {output.shape}, dtype: {output.dtype}")
        print(f"input shape: {input.shape}, dtype: {input.dtype}")
        print(
            f"weight shape: {self.weight.data.shape}, dtype: {self.weight.data.dtype}"
        )
        print(
            f"combo_position_ids shape: {combo_position_ids.shape}, dtype: {combo_position_ids.dtype}"
        )
        print(
            f"position_encoding shape: {position_encoding.shape}, dtype: {position_encoding.dtype}"
        )
        print(
            f"combo_tokens_type_ids shape: {combo_tokens_type_ids.shape}, dtype: {combo_tokens_type_ids.dtype}"
        )
        print(
            f"token_type_embedding shape: {token_type_embedding.shape}, dtype: {token_type_embedding.dtype}"
        )
        print(f"input_embedding_scalar: {input_embedding_scalar}")
        rtp_llm_ops.embedding_bert(
            output,
            input,
            self.weight.data,
            combo_position_ids,
            position_encoding,
            combo_tokens_type_ids,
            token_type_embedding,
            input_embedding_scalar,
        )
        print("embedding bert end")
        if self.config.tp_size > 1:
            m, n = output.shape
            output = all_gather(output, group=Group.TP)
            output = (
                output.reshape(self.config.tp_size, m, n)
                .transpose(0, 1)
                .contiguous()
                .reshape(m, -1)
            )
        return output
