import unittest

from google.protobuf.descriptor import FieldDescriptor

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (  # type: ignore
    BatchGenerateOutputsPB,
    ErrorCodePB,
    ErrorDetailsPB,
    GenerateConfigPB,
    RpcErrorPB,
)


class RpcErrorCodeProtoTest(unittest.TestCase):
    def test_invalid_params_is_declared_and_round_trips(self):
        self.assertEqual(ErrorCodePB.Value("INVALID_PARAMS"), 26)

        serialized = RpcErrorPB(
            error_code=ErrorCodePB.INVALID_PARAMS,
            error_message="invalid zero-prefill config",
        ).SerializeToString()
        parsed = RpcErrorPB.FromString(serialized)

        self.assertEqual(parsed.error_code, ErrorCodePB.INVALID_PARAMS)
        self.assertEqual(ErrorCodePB.Name(parsed.error_code), "INVALID_PARAMS")
        self.assertEqual(parsed.error_message, "invalid zero-prefill config")

    def test_execution_exception_is_declared_and_round_trips(self):
        self.assertEqual(ErrorCodePB.Value("EXECUTION_EXCEPTION"), 27)

        serialized = RpcErrorPB(
            error_code=ErrorCodePB.EXECUTION_EXCEPTION,
            error_message="handler threw",
        ).SerializeToString()
        parsed = RpcErrorPB.FromString(serialized)

        self.assertEqual(parsed.error_code, ErrorCodePB.EXECUTION_EXCEPTION)
        self.assertEqual(ErrorCodePB.Name(parsed.error_code), "EXECUTION_EXCEPTION")
        self.assertEqual(parsed.error_message, "handler threw")

    def test_service_unhealthy_has_unique_wire_ordinal_and_round_trips(self):
        self.assertEqual(ErrorCodePB.Value("SERVICE_UNHEALTHY"), 28)
        self.assertEqual(len(ErrorCodePB.values()), len(set(ErrorCodePB.values())))
        error = RpcErrorPB(
            error_code=ErrorCodePB.SERVICE_UNHEALTHY,
            error_message="serving device unavailable",
        )
        parsed = RpcErrorPB.FromString(error.SerializeToString())
        self.assertEqual(parsed.error_code, ErrorCodePB.SERVICE_UNHEALTHY)
        self.assertEqual(ErrorCodePB.Name(parsed.error_code), "SERVICE_UNHEALTHY")
        self.assertEqual(parsed.error_message, error.error_message)

    def test_capture_backpressure_has_unique_wire_ordinal_and_round_trips(self):
        self.assertEqual(ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE, 29)
        self.assertEqual(ErrorCodePB.SERVICE_UNHEALTHY, 28)
        self.assertEqual(len(ErrorCodePB.values()), len(set(ErrorCodePB.values())))
        for message in ("", "capture admission full"):
            with self.subTest(message=message):
                error = RpcErrorPB(
                    error_code=ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE,
                    error_message=message,
                )
                parsed = RpcErrorPB.FromString(error.SerializeToString())
                self.assertEqual(parsed.error_code, 29)
                self.assertEqual(
                    ErrorCodePB.Name(parsed.error_code),
                    "HIDDEN_STATE_CAPTURE_BACKPRESSURE",
                )
                self.assertEqual(parsed.error_message, message)

    def test_native_details_preserve_codes_without_wire_ordinal_translation(self):
        for code in (8107, 8215, 8216, 28, 29):
            with self.subTest(code=code):
                details = ErrorDetailsPB(error_code=code)
                parsed = ErrorDetailsPB.FromString(details.SerializeToString())
                self.assertEqual(parsed.error_code, code)

    def test_batch_backpressure_and_health_keep_distinct_wire_codes(self):
        response = BatchGenerateOutputsPB()
        codes = (29, 28, 29)
        for code in codes:
            response.results.add().error_info.error_code = code
        parsed = BatchGenerateOutputsPB.FromString(response.SerializeToString())
        self.assertEqual(
            [item.error_info.error_code for item in parsed.results], list(codes)
        )
        self.assertEqual(
            [ErrorCodePB.Name(item.error_info.error_code) for item in parsed.results],
            [
                "HIDDEN_STATE_CAPTURE_BACKPRESSURE",
                "SERVICE_UNHEALTHY",
                "HIDDEN_STATE_CAPTURE_BACKPRESSURE",
            ],
        )

    def test_capture_hidden_states_keeps_field_77_wire_encoding(self):
        field = GenerateConfigPB.DESCRIPTOR.fields_by_name["capture_hidden_states"]
        self.assertEqual(field.number, 77)
        self.assertEqual(field.type, FieldDescriptor.TYPE_BOOL)
        self.assertFalse(field.has_presence)
        for enabled, wire in ((False, b""), (True, b"\xe8\x04\x01")):
            with self.subTest(enabled=enabled):
                config = GenerateConfigPB(capture_hidden_states=enabled)
                self.assertEqual(config.SerializeToString(), wire)
                self.assertEqual(
                    GenerateConfigPB.FromString(wire).capture_hidden_states, enabled
                )

    def test_batch_health_rejections_keep_order_and_wire_code(self):
        response = BatchGenerateOutputsPB()
        for index in range(3):
            error = response.results.add().error_info
            error.error_code = ErrorCodePB.SERVICE_UNHEALTHY
            error.error_message = f"health reject item {index}"
        parsed = BatchGenerateOutputsPB.FromString(response.SerializeToString())
        self.assertEqual(len(parsed.results), 3)
        for index, result in enumerate(parsed.results):
            self.assertTrue(result.HasField("error_info"))
            self.assertFalse(result.HasField("final_output"))
            self.assertEqual(
                result.error_info.error_code, ErrorCodePB.SERVICE_UNHEALTHY
            )
            self.assertEqual(
                result.error_info.error_message, f"health reject item {index}"
            )

    def test_disk_cache_and_prefill_only_keep_independent_wire_fields(self):
        fields = GenerateConfigPB.DESCRIPTOR.fields_by_name
        self.assertEqual(fields["enable_disk_cache"].number, 74)
        self.assertEqual(fields["enable_disk_cache"].type, FieldDescriptor.TYPE_BOOL)
        self.assertEqual(fields["aux_info"].number, 75)
        self.assertEqual(
            fields["aux_info"].message_type.full_name, "google.protobuf.BoolValue"
        )
        self.assertEqual(fields["prefill_only"].number, 76)
        for disk_cache in (False, True):
            for prefill_only in (False, True):
                with self.subTest(disk_cache=disk_cache, prefill_only=prefill_only):
                    config = GenerateConfigPB(
                        max_new_tokens=0,
                        enable_disk_cache=disk_cache,
                        prefill_only=prefill_only,
                    )
                    parsed = GenerateConfigPB.FromString(config.SerializeToString())
                    self.assertEqual(parsed.enable_disk_cache, disk_cache)
                    self.assertEqual(parsed.prefill_only, prefill_only)
                    self.assertEqual(parsed.max_new_tokens, 0)

    def test_prefill_only_disambiguates_plain_max_new_tokens_wire_zero(self):
        max_new_tokens = GenerateConfigPB.DESCRIPTOR.fields_by_name["max_new_tokens"]
        self.assertEqual(max_new_tokens.number, 1)
        self.assertEqual(max_new_tokens.type, FieldDescriptor.TYPE_INT32)
        self.assertFalse(max_new_tokens.has_presence)

        prefill_only = GenerateConfigPB.DESCRIPTOR.fields_by_name["prefill_only"]
        self.assertEqual(prefill_only.number, 76)
        self.assertEqual(prefill_only.type, FieldDescriptor.TYPE_BOOL)

        bare_zero = GenerateConfigPB(max_new_tokens=0).SerializeToString()
        self.assertEqual(bare_zero, b"")
        parsed_bare_zero = GenerateConfigPB.FromString(bare_zero)
        self.assertEqual(parsed_bare_zero.max_new_tokens, 0)
        self.assertFalse(parsed_bare_zero.prefill_only)

        flagged_zero = GenerateConfigPB(
            max_new_tokens=0, prefill_only=True
        ).SerializeToString()
        self.assertEqual(flagged_zero, b"\xe0\x04\x01")
        parsed_flagged_zero = GenerateConfigPB.FromString(flagged_zero)
        self.assertEqual(parsed_flagged_zero.max_new_tokens, 0)
        self.assertTrue(parsed_flagged_zero.prefill_only)


if __name__ == "__main__":
    unittest.main()
