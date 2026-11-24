"""Zero-dependency Qwen2/Qwen4.5 BPE Tokenizer reading ``tokenizer.json`` directly.

Matches HuggingFace ``Qwen2Tokenizer`` / ``tokenizers`` BPE behavior for prompt
formatting, encoding, and streaming UTF-8 decoding without requiring external
``transformers`` or ``tokenizers`` packages.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


def _bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_TO_UNICODE: Dict[int, str] = _bytes_to_unicode()
_UNICODE_TO_BYTE: Dict[str, int] = {v: k for k, v in _BYTE_TO_UNICODE.items()}

_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch).startswith("M")


def _is_letter_or_mark(ch: str) -> bool:
    return unicodedata.category(ch)[:1] in ("L", "M")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


def _is_punct_other(ch: str) -> bool:
    return (not ch.isspace()) and (not _is_letter_or_mark(ch)) and (not _is_number(ch))


try:
    import regex as _regex_mod  # type: ignore[import-not-found]

    _QWEN2_REGEX = _regex_mod.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
except Exception:
    _regex_mod = None
    _QWEN2_REGEX = None


def _pretokenize_qwen2(text: str) -> List[str]:
    """Split text using the exact Qwen2 ``ByteLevel`` + ``Split`` pretokenizer rules."""
    if not text:
        return []
    if _QWEN2_REGEX is not None:
        return _QWEN2_REGEX.findall(text)

    out: List[str] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]

        # 1. English contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if ch == "'":
            low = text[i : i + 3].lower()
            matched_c = False
            for c in _CONTRACTIONS:
                if low.startswith(c):
                    out.append(text[i : i + len(c)])
                    i += len(c)
                    matched_c = True
                    break
            if matched_c:
                continue

        # 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if _is_letter_or_mark(ch):
            j = i + 1
            while j < n and _is_letter_or_mark(text[j]):
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if (
            ch not in ("\r", "\n")
            and not _is_letter(ch)
            and not _is_number(ch)
            and i + 1 < n
            and _is_letter_or_mark(text[i + 1])
        ):
            j = i + 2
            while j < n and _is_letter_or_mark(text[j]):
                j += 1
            out.append(text[i:j])
            i = j
            continue

        # 3. \p{N} (single digit/numeric character)
        if _is_number(ch):
            out.append(ch)
            i += 1
            continue

        # 4. ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*'
        if _is_punct_other(ch) or (ch == " " and i + 1 < n and _is_punct_other(text[i + 1])):
            j = i + (1 if ch == " " else 0)
            while j < n and _is_punct_other(text[j]):
                j += 1
            while j < n and text[j] in ("\r", "\n"):
                j += 1
            out.append(text[i:j])
            i = j
            continue

        # 5 & 6. Whitespace: '\s*[\r\n]+' | '\s+(?!\S)' | '\s+'
        if ch.isspace():
            j = i
            last_nl = -1
            while j < n and text[j].isspace():
                if text[j] in ("\r", "\n"):
                    last_nl = j
                j += 1
            if last_nl != -1:
                out.append(text[i : last_nl + 1])
                i = last_nl + 1
            elif j < n and (j - i) > 1:
                # Leave final space to attach cleanly to the next word/punct if applicable
                out.append(text[i : j - 1])
                i = j - 1
            else:
                out.append(text[i:j])
                i = j
            continue

        out.append(ch)
        i += 1
    return out


class Qwen2BPETokenizer:
    """Exact zero-dependency Qwen2/Qwen4.5 BPE tokenizer backed by ``tokenizer.json``."""

    def __init__(self, tokenizer_dir_or_json: Union[str, Path]) -> None:
        p = Path(tokenizer_dir_or_json)
        tj_path = p / "tokenizer.json" if p.is_dir() else p
        if not tj_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found at {tj_path}")
        tj = json.loads(tj_path.read_text(encoding="utf-8"))

        model = tj["model"]
        self.vocab: Dict[str, int] = {str(k): int(v) for k, v in model["vocab"].items()}
        self.id_to_token: Dict[int, str] = {int(v): str(k) for k, v in self.vocab.items()}

        self.added_tokens: Dict[str, int] = {}
        self.special_token_ids: set[int] = set()
        for item in tj.get("added_tokens", ()):
            tok = str(item["content"])
            tid = int(item["id"])
            self.added_tokens[tok] = tid
            self.id_to_token[tid] = tok
            if item.get("special", True):
                self.special_token_ids.add(tid)

        tc_path = tj_path.parent / "tokenizer_config.json"
        self.eos_token_id: int = 248044
        self.bos_token_id: Optional[int] = None
        self.pad_token_id: Optional[int] = None
        if tc_path.exists():
            tc = json.loads(tc_path.read_text(encoding="utf-8"))
            eos_tok = tc.get("eos_token")
            if isinstance(eos_tok, str) and eos_tok in self.added_tokens:
                self.eos_token_id = self.added_tokens[eos_tok]
            elif "eos_token_id" in tc and tc["eos_token_id"] is not None:
                self.eos_token_id = int(tc["eos_token_id"])
            bos_tok = tc.get("bos_token")
            if isinstance(bos_tok, str) and bos_tok in self.added_tokens:
                self.bos_token_id = self.added_tokens[bos_tok]
            pad_tok = tc.get("pad_token")
            if isinstance(pad_tok, str) and pad_tok in self.added_tokens:
                self.pad_token_id = self.added_tokens[pad_tok]

        self.im_start_id: int = self.added_tokens.get("<|im_start|>", 248045)
        self.im_end_id: int = self.added_tokens.get("<|im_end|>", 248046)
        self.stop_token_ids: Tuple[int, ...] = tuple(
            sorted({self.eos_token_id, self.im_end_id})
        )

        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        for idx, m in enumerate(model.get("merges", ())):
            if isinstance(m, str):
                parts = m.split(" ")
                if len(parts) == 2:
                    self.bpe_ranks[(parts[0], parts[1])] = idx
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                self.bpe_ranks[(str(m[0]), str(m[1]))] = idx

        if self.added_tokens:
            escaped = sorted(self.added_tokens.keys(), key=len, reverse=True)
            self._special_re = re.compile("(" + "|".join(re.escape(k) for k in escaped) + ")")
        else:
            self._special_re = None

        self.vocab_size = max(self.id_to_token.keys(), default=0) + 1

    @lru_cache(maxsize=65536)
    def _bpe(self, token: str) -> Tuple[str, ...]:
        if len(token) <= 1:
            return (token,)
        word: List[str] = list(token)
        while len(word) > 1:
            best_pair: Optional[Tuple[str, str]] = None
            best_rank = 1 << 60
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            first, second = best_pair
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return tuple(word)

    def _encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in _pretokenize_qwen2(text):
            u_piece = "".join(_BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
            for bpe_tok in self._bpe(u_piece):
                tid = self.vocab.get(bpe_tok)
                if tid is not None:
                    ids.append(tid)
                else:
                    for ch in bpe_tok:
                        if ch in self.vocab:
                            ids.append(self.vocab[ch])
        return ids

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        if self._special_re is None:
            return self._encode_ordinary(text)
        ids: List[int] = []
        for seg in self._special_re.split(text):
            if not seg:
                continue
            sp_id = self.added_tokens.get(seg)
            if sp_id is not None:
                ids.append(sp_id)
            else:
                ids.extend(self._encode_ordinary(seg))
        return ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        stop_at_eos: bool = False,
    ) -> str:
        raw_bytes = bytearray()
        chunks: List[str] = []

        def _flush_bytes() -> None:
            if raw_bytes:
                chunks.append(bytes(raw_bytes).decode("utf-8", errors="replace"))
                raw_bytes.clear()

        for tid_raw in token_ids:
            tid = int(tid_raw)
            if stop_at_eos and tid in self.stop_token_ids:
                break
            if tid in self.added_tokens.values():
                if skip_special_tokens:
                    continue
                _flush_bytes()
                chunks.append(self.id_to_token.get(tid, ""))
                continue
            tok_str = self.id_to_token.get(tid)
            if tok_str is None:
                continue
            for ch in tok_str:
                b = _UNICODE_TO_BYTE.get(ch)
                if b is not None:
                    raw_bytes.append(b)
                else:
                    _flush_bytes()
                    chunks.append(ch)
        _flush_bytes()
        return "".join(chunks)

    def format_chat_prompt(
        self,
        user_message: str,
        *,
        system_message: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> str:
        """Format prompt using Qwen4.5-Flash ``chat_template.jinja`` conventions."""
        parts: List[str] = []
        if system_message is not None and system_message.strip():
            parts.append(f"<|im_start|>system\n{system_message.strip()}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{user_message.strip()}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        if enable_thinking:
            parts.append("<think>\n")
        else:
            parts.append("<think>\n\n</think>\n\n")
        return "".join(parts)

    def encode_chat(
        self,
        user_message: str,
        *,
        system_message: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> List[int]:
        return self.encode(
            self.format_chat_prompt(
                user_message,
                system_message=system_message,
                enable_thinking=enable_thinking,
            )
        )


__all__ = ["Qwen2BPETokenizer"]
