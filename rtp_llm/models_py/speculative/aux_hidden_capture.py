"""Decoder feature capture through the existing MTP hidden-state getter."""

import torch


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
        self._aux_graph_hidden = None
        self._aux_graph_capacity = (
            max_generate_batch_size * (config.gen_num_per_cycle + 1) if layer_ids else 0
        )
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

    def begin_aux_hidden_capture(self, hidden, is_cuda_graph=False):
        if not self._aux_columns:
            return
        rows, width = hidden.shape
        width *= len(self._aux_columns)
        if is_cuda_graph:
            if rows > self._aux_graph_capacity:
                raise ValueError("aux capture exceeds the decode graph token bound")
            if self._aux_graph_hidden is None:
                self._aux_graph_hidden = hidden.new_empty(
                    (self._aux_graph_capacity, width)
                )
            self._aux_hidden = self._aux_graph_hidden[:rows]
        else:
            # Eager prompt storage follows the actual forward, not the maximum
            # admitted context batch. Graphs retain their own stable address.
            self._aux_hidden = hidden.new_empty((rows, width))

    def capture_aux_hidden(self, layer_id, hidden, residual=None):
        column = self._aux_columns.get(layer_id)
        if column is None:
            return
        rows, width = hidden.shape
        output = self.get_mtp_target_hidden_states(rows)
        target = output[:, column * width : (column + 1) * width]
        if residual is None:
            target.copy_(hidden)
        else:
            torch.add(hidden, residual, out=target)

    def get_mtp_target_hidden_states(self, num_tokens):
        return self._aux_rows(self._aux_hidden, num_tokens)

    def get_mtp_target_hidden_states_for_graph(self, num_tokens):
        # Replay does not execute Python forward or update _aux_hidden.
        return self._aux_rows(self._aux_graph_hidden, num_tokens)

    @staticmethod
    def _aux_rows(buffer, num_tokens):
        if buffer is None:
            return None
        if not 0 <= num_tokens <= buffer.shape[0]:
            raise ValueError(
                f"aux capture rows {num_tokens} exceed buffer capacity {buffer.shape[0]}"
            )
        return buffer[:num_tokens]
