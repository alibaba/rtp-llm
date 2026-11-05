import unittest

from google.protobuf.descriptor import FieldDescriptor

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (  # type: ignore
    ErrorCodePB,
    GenerateConfigPB,
    RpcErrorPB,
)


class RpcErrorCodeProtoTest(unittest.TestCase):
    def test_declared_error_codes_round_trip(self):
        for name, value, message in (
            ("INVALID_PARAMS", 26, "invalid zero-prefill config"),
            ("EXECUTION_EXCEPTION", 27, "handler threw"),
            ("HIDDEN_STATE_CAPTURE_BACKPRESSURE", 28, ""),
            ("HIDDEN_STATE_CAPTURE_BACKPRESSURE", 28, "capture admission full"),
        ):
            with self.subTest(name=name, message=message):
                self.assertEqual(ErrorCodePB.Value(name), value)
                serialized = RpcErrorPB(
                    error_code=value, error_message=message
                ).SerializeToString()
                parsed = RpcErrorPB.FromString(serialized)
                self.assertEqual(parsed.error_code, value)
                self.assertEqual(ErrorCodePB.Name(parsed.error_code), name)
                self.assertEqual(parsed.error_message, message)

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

    def test_capture_hidden_states_has_an_independent_opt_in_wire_field(self):
        fields = GenerateConfigPB.DESCRIPTOR.fields_by_name
        self.assertNotIn("export_hidden_states", fields)
        field = fields["capture_hidden_states"]
        self.assertEqual(field.number, 78)
        self.assertEqual(field.type, FieldDescriptor.TYPE_BOOL)
        self.assertFalse(GenerateConfigPB().capture_hidden_states)
        self.assertEqual(
            GenerateConfigPB(capture_hidden_states=True).SerializeToString(),
            b"\xf0\x04\x01",
        )
        for max_new_tokens in (0, 7):
            for enabled in (False, True):
                with self.subTest(max_new_tokens=max_new_tokens, enabled=enabled):
                    config = GenerateConfigPB(
                        max_new_tokens=max_new_tokens,
                        prefill_only=max_new_tokens == 0,
                        capture_hidden_states=enabled,
                    )
                    parsed = GenerateConfigPB.FromString(config.SerializeToString())
                    self.assertEqual(parsed.capture_hidden_states, enabled)
                    self.assertEqual(parsed.max_new_tokens, max_new_tokens)
                    self.assertEqual(parsed.prefill_only, max_new_tokens == 0)
                    self.assertFalse(parsed.return_hidden_states)
                    self.assertFalse(parsed.return_all_hidden_states)

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
