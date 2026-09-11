"""Decoder feature capture through the existing MTP hidden-state getter."""

from rtp_llm.utils.model_weight import W


class AuxHiddenCaptureMixin:
    _captures_aux_hidden = True

    def __init__(
        self, config, parallelism_config, weights, max_generate_batch_size, **kwargs
    ):
        super().__init__(
            config, parallelism_config, weights, max_generate_batch_size, **kwargs
        )
        layer_ids = (
            config.capture_aux_hidden_layer_ids or []
            if self._captures_aux_hidden
            else []
        )
        self._aux_columns = {layer: column for column, layer in enumerate(layer_ids)}
        self._aux_hidden = None
        if layer_ids:
            if layer_ids != sorted(set(layer_ids)) or not (
                0 <= layer_ids[0] <= layer_ids[-1] < config.num_layers
            ):
                raise ValueError(
                    "aux capture layers must be sorted, unique and in range"
                )
            if parallelism_config.prefill_cp_config.is_enabled():
                raise NotImplementedError(
                    "Qwen aux capture requires replicated prefill"
                )
            # ModelFactory supplies the admitted prefill token bound for fixed
            # buffers. Allocate once: both eager forward and graph replay write
            # this address, including when a prompt runs between two replays.
            capacity = max(
                config.moe_prefill_max_tokens_per_rank or config.max_seq_len,
                max_generate_batch_size * (config.gen_num_per_cycle + 1),
            )
            reference = weights.get_global_weight(W.embedding)
            self._aux_hidden = reference.new_empty(
                (capacity, len(layer_ids) * config.hidden_size)
            )

    def capture_aux_hidden(self, layer_id, hidden, residual=None):
        column = self._aux_columns.get(layer_id)
        if column is None:
            return
        rows, width = hidden.shape
        output = self.get_mtp_target_hidden_states(rows)
        output[:, column * width : (column + 1) * width].copy_(
            hidden if residual is None else hidden + residual
        )

    def get_mtp_target_hidden_states(self, num_tokens):
        if self._aux_hidden is None:
            return None
        if not 0 <= num_tokens <= self._aux_hidden.shape[0]:
            raise ValueError(
                f"aux capture rows {num_tokens} exceed fixed buffer capacity"
            )
        return self._aux_hidden[:num_tokens]
