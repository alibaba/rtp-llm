"""Unit tests for ``rtp_llm.dash_sc.dashscope_compat``."""

from __future__ import annotations

import logging
import struct
from unittest import TestCase, main

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.dash_sc.dashscope_compat import (
    DashScopeRequestExtras,
    DashScopeResponseEchoArgs,
    append_dashscope_response_extras,
    apply_dashscope_extras_to_generate_config,
    parse_dashscope_request_extras,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2


def _add_string_param(
    req: predict_v2_pb2.ModelInferRequest, name: str, value: str
) -> None:
    req.parameters[name].string_param = value


def _add_int64_param(
    req: predict_v2_pb2.ModelInferRequest, name: str, value: int
) -> None:
    req.parameters[name].int64_param = value


def _add_int32_input(
    req: predict_v2_pb2.ModelInferRequest, name: str, value: int
) -> None:
    inp = req.inputs.add()
    inp.name = name
    inp.datatype = "INT32"
    inp.shape.append(1)
    req.raw_input_contents.append(struct.pack("<i", int(value)))


class ParseDashScopeRequestExtrasTest(TestCase):
    def test_empty_request_yields_empty_extras(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        extras = parse_dashscope_request_extras(req, ())
        self.assertEqual(extras, DashScopeRequestExtras())

    def test_stop_param_string_list(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_string_param(req, "stop", '["END","FIN"]')
        extras = parse_dashscope_request_extras(req, None)
        self.assertEqual(extras.stop_strs, ("END", "FIN"))
        self.assertEqual(extras.unsupported, ())

    def test_stop_param_bare_string_coerced(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_string_param(req, "stop", '"DONE"')
        extras = parse_dashscope_request_extras(req, None)
        self.assertEqual(extras.stop_strs, ("DONE",))

    def test_stop_param_malformed_json_recorded(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_string_param(req, "stop", "not_json[")
        extras = parse_dashscope_request_extras(req, None)
        self.assertEqual(extras.stop_strs, ())
        self.assertEqual(len(extras.unsupported), 1)
        self.assertEqual(extras.unsupported[0][0], "stop")

    def test_stop_token_ids_flat_list_one_group(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_string_param(req, "stop_token_ids", "[1, 2, 3]")
        extras = parse_dashscope_request_extras(req, None)
        self.assertEqual(extras.extra_stop_token_groups, ((1, 2, 3),))

    def test_stop_token_ids_nested_list(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_string_param(req, "stop_token_ids", "[[1,2],[3]]")
        extras = parse_dashscope_request_extras(req, None)
        self.assertEqual(extras.extra_stop_token_groups, ((1, 2), (3,)))

    def test_stop_token_ids_invalid_shape_recorded(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_string_param(req, "stop_token_ids", '[{"foo":1}]')
        extras = parse_dashscope_request_extras(req, None)
        self.assertEqual(extras.extra_stop_token_groups, ())
        names = [u[0] for u in extras.unsupported]
        self.assertIn("stop_token_ids", names)

    def test_max_matched_token_num_negative_suppresses(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        extras = parse_dashscope_request_extras(
            req, [("x-ds-max-matched-token-num", "-1")]
        )
        self.assertTrue(extras.suppress_cached_token_num)

    def test_max_matched_token_num_positive_no_suppress(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        extras = parse_dashscope_request_extras(
            req, [("x-ds-max-matched-token-num", "10")]
        )
        self.assertFalse(extras.suppress_cached_token_num)

    def test_max_matched_token_num_non_int_recorded(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        extras = parse_dashscope_request_extras(
            req, [("x-ds-max-matched-token-num", "abc")]
        )
        self.assertFalse(extras.suppress_cached_token_num)
        names = [u[0] for u in extras.unsupported]
        self.assertIn("x-ds-max-matched-token-num", names)

    def test_max_matched_token_num_case_insensitive(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        extras = parse_dashscope_request_extras(
            req, [("X-DS-MAX-MATCHED-TOKEN-NUM", "0")]
        )
        self.assertTrue(extras.suppress_cached_token_num)

    def test_incremental_output_param(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_int64_param(req, "incremental_output", 1)
        extras = parse_dashscope_request_extras(req, None)
        self.assertEqual(extras.incremental_output, True)

    def test_scheduler_request_id_param(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_string_param(req, "scheduler_request_id", "sched-42")
        extras = parse_dashscope_request_extras(req, None)
        self.assertEqual(extras.scheduler_request_id, "sched-42")

    def test_scheduler_request_id_empty_string_treated_as_none(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_string_param(req, "scheduler_request_id", "")
        extras = parse_dashscope_request_extras(req, None)
        self.assertIsNone(extras.scheduler_request_id)

    def test_max_new_think_tokens_input_tensor(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_int32_input(req, "max_new_think_tokens", 256)
        extras = parse_dashscope_request_extras(req, None)
        self.assertEqual(extras.max_thinking_tokens, 256)

    def test_max_think_length_alias_for_max_new_think_tokens(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_int32_input(req, "max_think_length", 128)
        extras = parse_dashscope_request_extras(req, None)
        self.assertEqual(extras.max_thinking_tokens, 128)

    def test_logprobs_input_enables(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_int32_input(req, "logprobs", 1)
        _add_int32_input(req, "top_logprobs", 5)
        extras = parse_dashscope_request_extras(req, None)
        self.assertTrue(extras.enable_logprobs)
        self.assertEqual(extras.top_logprobs, 5)

    def test_logprobs_zero_does_not_enable(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_int32_input(req, "logprobs", 0)
        extras = parse_dashscope_request_extras(req, None)
        self.assertFalse(extras.enable_logprobs)

    def test_engine_unsupported_param_fields_recorded(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_string_param(req, "response_format", '{"type":"json_object"}')
        _add_string_param(req, "guided_json", '{"type":"object"}')
        _add_string_param(req, "logit_bias", '{"42": -10}')
        _add_int64_param(req, "context_cache_ttl", 60)
        extras = parse_dashscope_request_extras(req, None)
        names = sorted(u[0] for u in extras.unsupported)
        self.assertEqual(
            names,
            [
                "context_cache_ttl",
                "guided_json",
                "logit_bias",
                "response_format",
            ],
        )

    def test_unsupported_input_context_cache_ttl_recorded(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_int32_input(req, "context_cache_ttl", 30)
        extras = parse_dashscope_request_extras(req, None)
        names = [u[0] for u in extras.unsupported]
        self.assertIn("input.context_cache_ttl", names)


class ApplyDashScopeExtrasToGenerateConfigTest(TestCase):
    def test_stop_strs_union_with_existing_stop_words_str(self) -> None:
        gc = GenerateConfig(stop_words_str=["EXISTING"])
        extras = DashScopeRequestExtras(stop_strs=("END", "EXISTING"))
        apply_dashscope_extras_to_generate_config(gc, extras, request_log_tag="t")
        # set() collapses duplicates; both unique entries survive.
        self.assertEqual(set(gc.stop_words_str), {"EXISTING", "END"})

    def test_stop_token_groups_dedup(self) -> None:
        gc = GenerateConfig(stop_words_list=[[1, 2]])
        extras = DashScopeRequestExtras(extra_stop_token_groups=((1, 2), (3, 4)))
        apply_dashscope_extras_to_generate_config(gc, extras, request_log_tag="t")
        self.assertEqual(gc.stop_words_list, [[1, 2], [3, 4]])

    def test_enable_logprobs_sets_return_all_probs(self) -> None:
        gc = GenerateConfig()
        extras = DashScopeRequestExtras(enable_logprobs=True)
        apply_dashscope_extras_to_generate_config(gc, extras, request_log_tag="t")
        self.assertTrue(gc.return_all_probs)

    def test_max_thinking_tokens_passthrough(self) -> None:
        gc = GenerateConfig()
        extras = DashScopeRequestExtras(max_thinking_tokens=512)
        apply_dashscope_extras_to_generate_config(gc, extras, request_log_tag="t")
        self.assertEqual(gc.max_thinking_tokens, 512)

    def test_incremental_output_sets_return_incremental(self) -> None:
        gc = GenerateConfig()
        extras = DashScopeRequestExtras(incremental_output=True)
        apply_dashscope_extras_to_generate_config(gc, extras, request_log_tag="t")
        self.assertTrue(gc.return_incremental)

    def test_scheduler_request_id_sets_chat_id(self) -> None:
        gc = GenerateConfig()
        extras = DashScopeRequestExtras(scheduler_request_id="sched-77")
        apply_dashscope_extras_to_generate_config(gc, extras, request_log_tag="t")
        self.assertEqual(gc.chat_id, "sched-77")

    def test_unsupported_field_emits_warning_log(self) -> None:
        gc = GenerateConfig()
        extras = DashScopeRequestExtras(
            unsupported=(("response_format", "engine_unsupported"),)
        )
        with self.assertLogs(level=logging.WARNING) as cm:
            apply_dashscope_extras_to_generate_config(
                gc, extras, request_log_tag="tag-xyz"
            )
        joined = "\n".join(cm.output)
        self.assertIn("response_format", joined)
        self.assertIn("tag-xyz", joined)

    def test_empty_extras_is_noop(self) -> None:
        gc = GenerateConfig()
        before = gc.model_dump()
        apply_dashscope_extras_to_generate_config(
            gc, DashScopeRequestExtras(), request_log_tag="t"
        )
        self.assertEqual(gc.model_dump(), before)


class AppendDashScopeResponseExtrasTest(TestCase):
    def _new_infer(self) -> predict_v2_pb2.ModelInferResponse:
        infer = predict_v2_pb2.ModelInferResponse()
        # Pre-seed metric parameters as the codec does, to verify suppress
        # actually overrides them.
        infer.parameters["prompt_token_num"].int64_param = 100
        infer.parameters["prompt_cached_token_num"].int64_param = 30
        return infer

    def test_suppress_cached_overwrites_to_zero(self) -> None:
        infer = self._new_infer()
        echo = DashScopeResponseEchoArgs(suppress_cached_token_num=True)
        append_dashscope_response_extras(infer, echo)
        self.assertEqual(infer.parameters["prompt_cached_token_num"].int64_param, 0)
        # prompt_token_num must be untouched.
        self.assertEqual(infer.parameters["prompt_token_num"].int64_param, 100)

    def test_no_suppress_leaves_cached_unchanged(self) -> None:
        infer = self._new_infer()
        echo = DashScopeResponseEchoArgs(suppress_cached_token_num=False)
        append_dashscope_response_extras(infer, echo)
        self.assertEqual(infer.parameters["prompt_cached_token_num"].int64_param, 30)

    def test_string_echoes_set_when_provided(self) -> None:
        infer = predict_v2_pb2.ModelInferResponse()
        echo = DashScopeResponseEchoArgs(
            model_name="qwen3",
            instance_ip="10.0.0.5",
            hostname="host-x",
            scheduler_request_id="sched-7",
        )
        append_dashscope_response_extras(infer, echo)
        self.assertEqual(infer.parameters["model_name"].string_param, "qwen3")
        self.assertEqual(infer.parameters["model_instance_ip"].string_param, "10.0.0.5")
        self.assertEqual(infer.parameters["model_hostname"].string_param, "host-x")
        self.assertEqual(
            infer.parameters["scheduler_request_id"].string_param, "sched-7"
        )

    def test_empty_echo_writes_nothing(self) -> None:
        infer = predict_v2_pb2.ModelInferResponse()
        append_dashscope_response_extras(infer, DashScopeResponseEchoArgs())
        # No side effects on parameters when nothing supplied.
        self.assertEqual(len(infer.parameters), 0)


if __name__ == "__main__":
    main()
