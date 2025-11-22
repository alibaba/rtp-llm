import torch

from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.model_factory_register import register_model
from rtp_llm.utils.base_model_datatypes import GenerateOutput
from rtp_llm.models.bloom import Bloom


class SGPTBloom(Bloom):
    @torch.no_grad()
    def generate_hidden_states_stream(self, input_token_ids: torch.IntTensor):
        assert (
            self.weight is not None
        ), "Please call load() first to initialize weights."
        input_token_ids_np = input_token_ids.cpu().numpy()
        batch_size = len(input_token_ids_np)
        eos_token_id = self.config.special_tokens.eos_token_id
        input_lengths = torch.IntTensor(
            [len(v[v != eos_token_id]) for v in input_token_ids_np]
        )
        input_token_ids = input_token_ids.type(torch.int32).to(self.device)
        input_lengths = input_lengths.type(torch.int32).to(self.device)
        max_input_length = input_token_ids.shape[-1]
        gen_length = 1
        beam_width = 1
        max_seq_length = max_input_length + gen_length
        memory_length = max_seq_length
        device = self.device
        # Since tril() doesn't support bf16 dtype, we create of bool type and then cast it to dtype.
        attention_mask = (
            torch.ones(
                (max_input_length, max_input_length), dtype=torch.bool, device=device
            )
            .tril()
            .unsqueeze(0)
        )
        attention_mask = attention_mask.tile(input_token_ids.shape[0], 1, 1).to(
            self.dtype
        )
        for b, input_length in enumerate(input_lengths):
            attention_mask[b, input_length:, ...] = 0
        if g_parallel_info.is_pp_first:
            # Prepare input tensors of decoder.
            input_embeds = self.word_embedding(input_token_ids)
            if self.position_encoding is not None:
                position_ids = torch.arange(
                    0, max_input_length, dtype=torch.int, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, max_input_length)
                input_embeds += self.position_encoding(position_ids)
            if self.pre_decoder_layernorm is not None:
                input_embeds = self.pre_decoder_layernorm(input_embeds)
        else:
            # Dummy input_embeds
            input_embeds = torch.empty(
                size=(
                    batch_size * beam_width,
                    max_input_length,
                    self.context_decoder.hidden_size,
                ),
                dtype=self.context_decoder.dtype,
                device=device,
            )
        hidden_states, _, _, _ = self.context_decoder.forward(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            memory_length=memory_length,
            linear_bias_slopes=self.linear_bias_slopes,
        )
        hidden_states = self.post_decoder_layernorm(hidden_states)  # type: ignore
        yield GenerateOutput(
            hidden_states,
            None,
            input_token_ids,
            torch.ones_like(input_lengths).bool(),
            [{}] * input_lengths.shape[0],
        )

    @torch.no_grad()
    def generate_stream(
        self, input_token_ids, input_lengths, generate_config  # type: ignore
    ):
        return self.generate_hidden_states_stream(input_token_ids=input_token_ids)


register_model("sgpt_bloom", SGPTBloom)
