"""Raw-token byte decoding at the third-party tokenizer boundary."""

import json
from collections.abc import Callable, Mapping
from typing import Optional

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast


def _byte_level_decoder() -> dict[str, int]:
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            characters.append(256 + extra)
            extra += 1
    return {chr(character): value for character, value in zip(characters, byte_values)}


class TokenBytesDecoder:
    """Resolve optional remote-code capabilities once, outside renderer loops."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self._tokenizer = tokenizer
        self._byte_decoder: Optional[Mapping[str, int]] = None
        self._raw_decoder: Optional[Callable[[int], bytes]] = None
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            backend_decoder = tokenizer.backend_tokenizer.decoder
            if backend_decoder is not None:
                state = json.loads(backend_decoder.__getstate__())
                decoders = state.get("decoders", [state])
                if any(decoder["type"] == "ByteLevel" for decoder in decoders):
                    self._byte_decoder = _byte_level_decoder()
        else:
            # Remote-code tokenizers have no common raw-byte contract. Keep
            # their optional capabilities here; known HF APIs use direct access.
            byte_decoder = getattr(tokenizer, "byte_decoder", None)
            if isinstance(byte_decoder, Mapping):
                self._byte_decoder = byte_decoder
            for name in ("token_id_to_bytes", "decode_single_token_bytes"):
                method = getattr(tokenizer, name, None)
                if callable(method):
                    self._raw_decoder = method
                    break

    def decode(self, token_id: int) -> Optional[bytes]:
        """Return exact bytes, or None if the backend cannot expose them."""
        if self._raw_decoder is not None:
            try:
                raw_bytes = self._raw_decoder(token_id)
            except (KeyError, IndexError):
                raw_bytes = None
            if isinstance(raw_bytes, (bytes, bytearray, memoryview)):
                return bytes(raw_bytes)
        piece = self._tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(piece, bytes):
            return piece
        if not isinstance(piece, str):
            return None
        if len(piece) == 6 and piece.startswith(("<0x", "<0X")) and piece.endswith(">"):
            try:
                return bytes([int(piece[3:5], 16)])
            except ValueError:
                return None
        if self._byte_decoder is not None:
            values = bytearray()
            for character in piece:
                value = self._byte_decoder.get(character)
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 0 <= value <= 255
                ):
                    return None
                values.append(value)
            return bytes(values)
        return None
