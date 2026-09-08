"""End-to-end test for the MiMo V2.5 model (48 layers, 256 MoE experts, FP8).

TP must be exactly 4: the FP8 checkpoint stores fused QKV as four native slabs and the
weight loader refuses anything else (``mimo_v25_weight.py``: ``assert tp ==
QKV_QUANT_SHARDS``), because slabs cannot be merged or re-split inside the FP8 domain.

Usage:
    CHECKPOINT_PATH=/path/to/MiMo-V2.5 TP_SIZE=4 REUSE_CACHE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
        python -m rtp_llm.test.model_test.test_mimo_v25
"""

import json
import logging
import os
import sys
import unittest

from rtp_llm.test.utils.maga_server_manager import MagaServerManager

# force=True is what makes this take effect at all. Importing rtp_llm above pulls in
# torch_patch, which emits a module-level logging.warning when torch is not the validated
# version; that implicitly runs logging.basicConfig() and leaves root at WARNING. Without
# force, this call would silently return and every logging.info below would be dropped.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def _find_checkpoint() -> str:
    path = os.environ.get("CHECKPOINT_PATH", "")
    if path and os.path.isfile(os.path.join(path, "config.json")):
        return path
    raise FileNotFoundError(
        "MiMo V2.5 checkpoint not found. Set CHECKPOINT_PATH to a directory "
        f"containing config.json (got {path!r})."
    )


# ---------------------------------------------------------------------------
# Long-context prompt
#
# MiMo V2.5 is hybrid: 9 global-attention layers plus 39 sliding-window layers with
# sliding_window=128. A prompt shorter than the window leaves every interesting path
# untested -- the window never masks anything, and a sequence shorter than the KV page
# size fits in one page so the block table is never indexed past entry 0. The prompt
# below crosses both boundaries.
# ---------------------------------------------------------------------------
SLIDING_WINDOW = 128  # config.json: sliding_window
KV_PAGE_SIZE = 64  # --seq_size_per_block default

_NEEDLE_CODE = "ZEBRA-7741"
_FILLER_LINES = 48

_FILLER = (
    "the maintenance crew inspects the turbine housing on a fixed schedule.",
    "coolant pressure is logged twice per shift by the on-duty technician.",
    "the vibration sensor on bearing three was replaced during the last outage.",
    "lubricant samples are sent to the laboratory for spectrographic analysis.",
    "the control room keeps a paper backup of every automated alarm.",
    "spare gaskets are stored in the west annex on the second shelf.",
    "the intake filter differential is trending slightly upward this quarter.",
    "grounding straps were re-torqued after the seismic retrofit.",
    "the auxiliary generator is exercised under load once per month.",
    "condensate traps are blown down before each cold start.",
    "the calibration certificate for the flow meter expires next spring.",
    "thermal imaging of the switchgear found no hot spots this cycle.",
)

# Keep the prefix-cache probe independent from the long-context warm-up requests.
# The block key is chained, so changing the first token is enough to invalidate all
# following blocks; distinct filler text also makes the separation obvious in logs.
_PROBE_FILLER = (
    "probe record alpha verifies an isolated cache namespace for this request.",
    "probe record beta stores a calibration value unrelated to equipment maintenance.",
    "probe record gamma tracks a synthetic payload boundary for allocator testing.",
    "probe record delta carries a unique sentence used only by the cache probe.",
    "probe record epsilon confirms that this prompt has no maintenance-log prefix.",
    "probe record zeta marks a deterministic token stream for repeated requests.",
    "probe record eta checks that local device blocks can be found again later.",
    "probe record theta keeps the probe longer than one sliding-attention window.",
    "probe record iota exercises a second page without sharing prior test content.",
    "probe record kappa supplies stable text for exact prefix matching.",
    "probe record lambda ends with a deliberately unrelated synthetic statement.",
    "probe record mu closes the isolated cache-prefix fixture for this test.",
)


def _build_long_prompt(with_question: bool = True) -> str:
    """A maintenance log with the answer planted near the very beginning.

    The needle sits in the second line; the question is ~900 tokens later, far outside
    the 128-token window of the final query token. Only the global attention layers can
    carry it that far, so recalling it means GA and SWA layers really are behaving
    differently.

    ``with_question=False`` leaves the log unfinished instead. The raw completion
    endpoint applies no chat template, so an instruction-shaped ending makes the model
    emit a stop token as its very first output -- measuring the prompt format, not the
    decode path. An unfinished log invites plain continuation.
    """
    lines = [
        "You are reading an equipment maintenance log. Read it carefully.",
        f"Note 00: the emergency override code for this facility is {_NEEDLE_CODE}.",
    ]
    for i in range(1, _FILLER_LINES + 1):
        lines.append(f"Note {i:02d}: {_FILLER[(i - 1) % len(_FILLER)]}")
    if with_question:
        lines.append(
            "Question: what is the emergency override code recorded in the log above? "
            "Reply with the code only."
        )
    else:
        lines.append(f"Note {_FILLER_LINES + 1:02d}:")
    return "\n".join(lines)


def _build_prefix_probe_prompt() -> str:
    """Build a long prompt whose prefix is unrelated to every warm-up test."""
    lines = [
        "CACHE_PROBE_ISOLATED_PREFIX_7F3B: use this synthetic record only.",
    ]
    for i in range(_FILLER_LINES * 2):
        lines.append(f"Probe {i:02d}: {_PROBE_FILLER[i % len(_PROBE_FILLER)]}")
    lines.append("Probe question: emit one token after this isolated prefix.")
    return "\n".join(lines)


def _decode_body(response_text) -> dict:
    if isinstance(response_text, list):
        response_text = b"".join(response_text).decode("utf-8", errors="replace")
    return json.loads(response_text)


class TestMiMoV25E2E(unittest.TestCase):
    """End-to-end smoke test for MiMo V2.5 via the RTP-LLM server."""

    _server = None
    _ckpt_path = None
    _reuse_cache_enabled = False

    @classmethod
    def setUpClass(cls):
        cls._ckpt_path = _find_checkpoint()
        logging.info(f"Using checkpoint: {cls._ckpt_path}")

        # tp_size 4 is the only value the FP8 checkpoint loads at, see module docstring.
        # max_seq_len must hold the long-context prompt below plus its output.
        tp_size = int(os.environ.get("TP_SIZE", "4"))
        reuse_cache_arg = os.environ.get("REUSE_CACHE", "1")
        cls._reuse_cache_enabled = reuse_cache_arg.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        smoke_args = (
            f"--model_type mimo_v25 "
            f"--checkpoint_path {cls._ckpt_path} "
            f"--tokenizer_path {cls._ckpt_path} "
            f"--tp_size {tp_size} "
            f"--world_size {tp_size} "
            f"--max_seq_len 2048 "
            f"--concurrency_limit 1 "
            f"--reuse_cache {reuse_cache_arg}"
        )

        cls._server = MagaServerManager(
            process_file_name="test_mimo_v25.log",
            smoke_args_str=smoke_args,
        )

        logging.info("Starting MiMo V2.5 server (this may take several minutes)...")
        if not cls._server.start_server(timeout=1600):
            cls._server.print_process_log()
            raise RuntimeError(
                "MiMo V2.5 server failed to start. See test_mimo_v25.log."
            )
        logging.info(f"Server started on port {cls._server.port}")

    @classmethod
    def tearDownClass(cls):
        if cls._server is not None:
            logging.info("Stopping server...")
            cls._server.stop_server()

    def test_text_generation_basic(self):
        """Send a simple prompt and verify non-empty generation output."""
        query = {
            "prompt": "The capital of France is",
            "generate_config": {
                "max_new_tokens": 32,
                "top_k": 1,
                "temperature": 0.0,
            },
        }

        success, response_text = self._server.visit(
            query=query, retry_times=3, endpoint="/"
        )
        self.assertTrue(success, "Server request failed after retries")
        self.assertIsNotNone(response_text, "Response is None")

        resp = _decode_body(response_text)
        generated = resp.get("response", "")
        logging.info(f"Generated text: {generated}")

        self.assertIsInstance(generated, str)
        self.assertTrue(
            len(generated.strip()) > 0,
            f"Generated text is empty. Full response: {resp}",
        )

    def test_openai_chat_completion(self):
        """Send a chat completion request via /v1/chat/completions."""
        query = {
            "model": "mimo_v25",
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "max_tokens": 32,
            "temperature": 0.0,
        }

        success, response_text = self._server.visit(
            query=query, retry_times=3, endpoint="/v1/chat/completions"
        )
        self.assertTrue(success, "OpenAI chat request failed after retries")
        self.assertIsNotNone(response_text, "Response is None")

        resp = _decode_body(response_text)
        choices = resp.get("choices", [])
        self.assertTrue(len(choices) > 0, f"No choices in response: {resp}")

        content = choices[0].get("message", {}).get("content", "")
        logging.info(f"Chat content: {content}")
        self.assertTrue(
            len(content.strip()) > 0,
            f"Chat content is empty. Full response: {resp}",
        )

    def test_long_context_needle_recall(self):
        """Recall a fact planted before the sliding window, over a multi-page KV cache.

        Exercises three things the short prompts above cannot reach:
          - all 39 SWA layers actually mask, so ``window_left`` is verified rather than
            assumed (a prompt under 128 tokens masks nothing);
          - the sequence spans several KV pages, so the block table is indexed past
            entry 0 and per-group page tables have to be right;
          - the needle is outside the window of the final query token, so only the 9
            global-attention layers can carry it -- GA and SWA must differ.
        """
        query = {
            "model": "mimo_v25",
            "messages": [{"role": "user", "content": _build_long_prompt()}],
            "max_tokens": 96,
            "temperature": 0.0,
        }

        success, response_text = self._server.visit(
            query=query, retry_times=3, endpoint="/v1/chat/completions"
        )
        self.assertTrue(success, "Long-context chat request failed after retries")
        self.assertIsNotNone(response_text, "Response is None")

        resp = _decode_body(response_text)
        choices = resp.get("choices", [])
        self.assertTrue(len(choices) > 0, f"No choices in response: {resp}")
        content = choices[0].get("message", {}).get("content", "")
        prompt_tokens = resp.get("usage", {}).get("prompt_tokens", 0)

        logging.info(f"Long-context prompt_tokens={prompt_tokens} content={content!r}")

        # Without this the test silently degrades into another short-prompt case and
        # stops covering the window at all.
        self.assertGreater(
            prompt_tokens,
            SLIDING_WINDOW,
            f"prompt is only {prompt_tokens} tokens, which does not exceed "
            f"sliding_window={SLIDING_WINDOW}; lengthen _FILLER_LINES so the window "
            f"and multi-page paths are actually exercised",
        )
        self.assertTrue(
            len(content.strip()) > 0,
            f"Long-context content is empty. Full response: {resp}",
        )
        self.assertIn(
            _NEEDLE_CODE,
            content,
            f"Model did not recall {_NEEDLE_CODE} planted {prompt_tokens} tokens back, "
            f"outside the {SLIDING_WINDOW}-token window. Global-attention layers or the "
            f"window mask are likely wrong. Got: {content!r}",
        )

    def test_long_prompt_decode_crosses_kv_page(self):
        """Generate enough tokens from a long prompt to allocate a fresh KV page.

        Asking for KV_PAGE_SIZE new tokens guarantees the sequence length crosses a
        multiple of the page size mid-generation, which forces an incremental block
        allocation -- a new physical block id in every KV cache group -- while decoding.
        That is a different path from prefill-time allocation.

        ``min_new_tokens`` is what makes that guarantee hold: it suppresses both the EOS
        and stop-word checks until the count is reached (``GenerateStream.cc``:
        ``seqLength() >= min_new_tokens + inputLength()``). Without it the model is free
        to stop on its first token and the page boundary is never reached.
        """
        query = {
            "prompt": _build_long_prompt(with_question=False),
            "generate_config": {
                "max_new_tokens": KV_PAGE_SIZE,
                "min_new_tokens": KV_PAGE_SIZE,
                "top_k": 1,
                "temperature": 0.0,
            },
        }

        success, response_text = self._server.visit(
            query=query, retry_times=3, endpoint="/"
        )
        self.assertTrue(success, "Long-prompt generation failed after retries")
        self.assertIsNotNone(response_text, "Response is None")

        resp = _decode_body(response_text)
        generated = resp.get("response", "")
        aux = resp.get("aux_info", {})
        input_len = aux.get("input_len", 0)
        output_len = aux.get("output_len", 0)

        logging.info(f"Long-prompt input_len={input_len} output_len={output_len}")

        self.assertGreater(
            input_len,
            SLIDING_WINDOW,
            f"input_len={input_len} does not exceed sliding_window={SLIDING_WINDOW}; "
            f"lengthen _FILLER_LINES",
        )
        # >= KV_PAGE_SIZE new tokens is what guarantees the sequence length crossed a
        # multiple of the page size, i.e. that decode really did allocate a new block.
        self.assertGreaterEqual(
            output_len,
            KV_PAGE_SIZE,
            f"only {output_len} tokens generated, so the sequence may never have "
            f"crossed a {KV_PAGE_SIZE}-token page boundary; min_new_tokens should have "
            f"forced {KV_PAGE_SIZE}. Full response: {resp}",
        )
        self.assertTrue(
            len(generated.strip()) > 0,
            f"Generated text is empty. Full response: {resp}",
        )

        # Non-empty alone would pass on garbage. The log rotates through _FILLER with
        # period len(_FILLER), so completing the unfinished note correctly means
        # attending back len(_FILLER) lines -- roughly 200 tokens, well past the
        # 128-token window. Only the global-attention layers reach that far, and here
        # they have to do it during *decode*, across the page boundary.
        expected_next = _FILLER[_FILLER_LINES % len(_FILLER)]
        probe = " ".join(expected_next.split()[:5])
        self.assertIn(
            probe,
            generated,
            f"continuation did not follow the {len(_FILLER)}-line rotation of the log. "
            f"Completing 'Note {_FILLER_LINES + 1:02d}:' requires attending ~200 tokens "
            f"back, outside the {SLIDING_WINDOW}-token window, so this points at the "
            f"global-attention layers or at decode after a page boundary. "
            f"Expected to find {probe!r}, got: {generated[:200]!r}",
        )

    def test_prefix_cache_reuse(self):
        """Verify a cold request misses and an identical follow-up reuses its prefix."""
        query = {
            # This prompt intentionally does not share a prefix with the long-context
            # warm-up tests, so the first request is a genuine cold-cache observation.
            "prompt": _build_prefix_probe_prompt(),
            "generate_config": {
                "max_new_tokens": 1,
                "min_new_tokens": 1,
                "top_k": 1,
                "temperature": 0.0,
            },
        }

        success, response_text = self._server.visit(
            query=query, retry_times=3, endpoint="/"
        )
        self.assertTrue(success, "First prefix-cache request failed after retries")
        self.assertIsNotNone(response_text, "First prefix-cache response is None")
        first = _decode_body(response_text)

        success, response_text = self._server.visit(
            query=query, retry_times=3, endpoint="/"
        )
        self.assertTrue(success, "Second prefix-cache request failed after retries")
        self.assertIsNotNone(response_text, "Second prefix-cache response is None")
        second = _decode_body(response_text)

        first_aux = first.get("aux_info", {}) or {}
        second_aux = second.get("aux_info", {}) or {}
        first_reuse = int(first_aux.get("reuse_len", 0) or 0)
        second_reuse = int(second_aux.get("reuse_len", 0) or 0)
        logging.info(
            "Prefix-cache probe: first_reuse_len=%d second_reuse_len=%d "
            "first_aux=%s second_aux=%s",
            first_reuse,
            second_reuse,
            first_aux,
            second_aux,
        )

        if self._reuse_cache_enabled:
            self.assertEqual(
                first_reuse,
                0,
                "The isolated prefix probe was already warm before its first request. "
                f"First aux_info: {first_aux}",
            )
            self.assertGreater(
                second_reuse,
                0,
                "REUSE_CACHE is enabled, but the repeated prompt reported "
                f"reuse_len=0. First aux_info: {first_aux}; second aux_info: {second_aux}",
            )


if __name__ == "__main__":
    try:
        logging.info(f"Checkpoint found: {_find_checkpoint()}")
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    unittest.main(verbosity=2)
