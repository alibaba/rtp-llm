"""Coverage for ``plan_dash_sc_thinking``'s input-tail dispatch.

Five priority-ordered cases (DashLLM-aligned + one RTP-only divergence):

  1. tail == ``<think>\\n\\n</think>\\n\\n`` -> ``thinking=False`` (think_mode=0)
  2. tail == ``<think>\\n``                   -> ``tail_full_bos`` (echo full bos)
  3. tail == ``<think>``                     -> ``tail_single_bos`` (echo single)
  4. nothing of the above + Auto             -> ``auto_append_single_bos`` (append + echo single)
  5. nothing of the above + Force            -> ``force_append_full_bos`` (append + echo full bos)

DashLLM has a 6th case "``<think>\\n`` within last 4096 tokens -> no echo, no
append" that RTP intentionally drops: with ``in_think_mode=True`` forcing the
C++ logits processor into thinking, relying on the model to spontaneously
re-emit ``<think>\\n`` is unreliable and was the root cause of MRCR
qid 40/233/300/555/643 empty-content failures. RTP falls those inputs through
to case 4/5 instead so the dashscope-side parser always has a ``<think>``
anchor in the streamed output.

A regression case guards against treating ``<think>\\n\\n</think>\\n\\n`` buried
in multi-turn history as case 1 — the tail-suffix check exists precisely so a
new turn whose tail is ``<think>\\n`` keeps phase-1 reasoning even when an
earlier assistant turn had an empty think block.
"""

from __future__ import annotations

import unittest

from rtp_llm.dash_sc.think import (
    THINK_MODE_AUTO,
    THINK_MODE_FORCE,
    DashScThinkConfig,
    plan_dash_sc_thinking,
)

# DSV4 token shapes used throughout (mirror mrcr_smoke_test config so future
# tokenizer drift only needs to be tracked in one place).
_BOS = (128821, 201)  # encode("<think>\n")
_END = (128822, 271)  # encode("</think>\n\n")
_EOS = (201, 128822, 271)  # encode("\n</think>\n\n")
_EMPTY = (128821, 271, 128822, 271)  # encode("<think>\n\n</think>\n\n")


def _cfg(mode: str = THINK_MODE_AUTO) -> DashScThinkConfig:
    return DashScThinkConfig(
        enabled=True,
        mode=mode,
        bos_tokens=_BOS,
        end_think_token_ids=_END,
        eos_tokens=_EOS,
        empty_tokens=_EMPTY,
        is_deepseek_v4=True,
        dsv4_abort_token_id=1,
    )


def _plan(ids, *, mode=THINK_MODE_AUTO, enable_thinking=True, max_new_think_tokens=128):
    return plan_dash_sc_thinking(
        ids,
        think_config=_cfg(mode=mode),
        enable_thinking=enable_thinking,
        max_new_think_tokens=max_new_think_tokens,
    )


class PlanDashScThinkingDispatchTest(unittest.TestCase):
    """Each case in the spec table maps to a dedicated test for grep-ability."""

    def test_case1_tail_empty_tokens_disables_thinking(self) -> None:
        plan = _plan([100, 101] + list(_EMPTY))
        self.assertFalse(plan.thinking)
        self.assertEqual(plan.reason, "empty_think_present_tail")

    def test_case1_history_empty_tokens_does_not_disable_when_tail_is_bos(
        self,
    ) -> None:
        # Multi-turn regression: an earlier assistant turn produced an empty
        # ``<think>\n\n</think>\n\n`` block; the new turn ends with the full
        # ``<think>\n`` BOS. DashLLM treats this as case 2 (tail_full_bos);
        # the old ``contains_sublist`` shortcut wrongly disabled thinking.
        ids = [10, 11] + list(_EMPTY) + [200, 201] + list(_BOS)
        plan = _plan(ids)
        self.assertTrue(plan.thinking)
        self.assertEqual(plan.reason, "tail_full_bos")
        self.assertEqual(tuple(plan.echo_prefix_ids), _BOS)

    def test_case2_tail_full_bos_echoes_full_bos(self) -> None:
        plan = _plan([42, 43] + list(_BOS))
        self.assertTrue(plan.thinking)
        self.assertEqual(plan.reason, "tail_full_bos")
        self.assertEqual(tuple(plan.echo_prefix_ids), _BOS)
        self.assertEqual(plan.prompt_append_len, 0)

    def test_case3_tail_single_bos_echoes_single_token(self) -> None:
        plan = _plan([7, 8, _BOS[0]])
        self.assertTrue(plan.thinking)
        self.assertEqual(plan.reason, "tail_single_bos")
        self.assertEqual(plan.echo_prefix_ids, [_BOS[0]])
        self.assertEqual(plan.prompt_append_len, 0)

    def test_dropped_dashllm_4096_branch_falls_through_to_auto_append(
        self,
    ) -> None:
        # DashLLM's ``bos_inside_prompt`` branch is intentionally absent in
        # RTP. Inputs that historically triggered it (bos somewhere in the
        # last 4096 tokens but not at the tail) MUST fall through to the
        # default Auto/Force append so the dashscope-side parser still gets
        # a ``<think>`` anchor in the streamed output.
        ids = [9, 10] + list(_BOS) + [11, 12, 13, 14, 15]
        plan = _plan(ids, mode=THINK_MODE_AUTO)
        self.assertTrue(plan.thinking)
        self.assertEqual(plan.reason, "auto_append_single_bos")
        self.assertEqual(plan.echo_prefix_ids, [_BOS[0]])
        self.assertEqual(plan.prompt_append_len, 1)
        self.assertEqual(plan.input_ids[-1], _BOS[0])

        force_plan = _plan(ids, mode=THINK_MODE_FORCE)
        self.assertEqual(force_plan.reason, "force_append_full_bos")
        self.assertEqual(tuple(force_plan.echo_prefix_ids), _BOS)
        self.assertEqual(force_plan.prompt_append_len, len(_BOS))

    def test_case4_auto_appends_single_bos(self) -> None:
        plan = _plan([5, 6, 7], mode=THINK_MODE_AUTO)
        self.assertTrue(plan.thinking)
        self.assertEqual(plan.reason, "auto_append_single_bos")
        self.assertEqual(plan.echo_prefix_ids, [_BOS[0]])
        self.assertEqual(plan.prompt_append_len, 1)
        self.assertEqual(plan.input_ids[-1], _BOS[0])

    def test_case5_force_appends_full_bos(self) -> None:
        plan = _plan([5, 6, 7], mode=THINK_MODE_FORCE)
        self.assertTrue(plan.thinking)
        self.assertEqual(plan.reason, "force_append_full_bos")
        self.assertEqual(tuple(plan.echo_prefix_ids), _BOS)
        self.assertEqual(plan.prompt_append_len, len(_BOS))
        self.assertEqual(tuple(plan.input_ids[-len(_BOS) :]), _BOS)


class PlanDashScThinkingGuardsTest(unittest.TestCase):
    def test_enable_thinking_false_short_circuits(self) -> None:
        plan = _plan([1, 2] + list(_BOS), enable_thinking=False)
        self.assertFalse(plan.thinking)
        self.assertEqual(plan.reason, "enable_thinking_false")

    def test_max_new_think_tokens_zero_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            _plan([1, 2] + list(_BOS), max_new_think_tokens=0)

    def test_disabled_config_returns_disabled_plan(self) -> None:
        plan = plan_dash_sc_thinking(
            [1, 2, 3],
            think_config=DashScThinkConfig.disabled(),
            enable_thinking=True,
            max_new_think_tokens=128,
        )
        self.assertFalse(plan.thinking)
        self.assertEqual(plan.reason, "think_config_disabled")


if __name__ == "__main__":
    unittest.main()
