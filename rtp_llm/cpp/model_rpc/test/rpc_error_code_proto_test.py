import unittest

from google.protobuf.descriptor import FieldDescriptor

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (  # type: ignore
    ErrorCodePB,
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

    def test_prefill_only_disambiguates_plain_max_new_tokens_wire_zero(self):
        max_new_tokens = GenerateConfigPB.DESCRIPTOR.fields_by_name["max_new_tokens"]
        self.assertEqual(max_new_tokens.number, 1)
        self.assertEqual(max_new_tokens.type, FieldDescriptor.TYPE_INT32)
        self.assertFalse(max_new_tokens.has_presence)

        prefill_only = GenerateConfigPB.DESCRIPTOR.fields_by_name["prefill_only"]
        self.assertEqual(prefill_only.number, 75)
        self.assertEqual(prefill_only.type, FieldDescriptor.TYPE_BOOL)

        bare_zero = GenerateConfigPB(max_new_tokens=0).SerializeToString()
        self.assertEqual(bare_zero, b"")
        parsed_bare_zero = GenerateConfigPB.FromString(bare_zero)
        self.assertEqual(parsed_bare_zero.max_new_tokens, 0)
        self.assertFalse(parsed_bare_zero.prefill_only)

        flagged_zero = GenerateConfigPB(
            max_new_tokens=0, prefill_only=True
        ).SerializeToString()
        self.assertEqual(flagged_zero, b"\xd8\x04\x01")
        parsed_flagged_zero = GenerateConfigPB.FromString(flagged_zero)
        self.assertEqual(parsed_flagged_zero.max_new_tokens, 0)
        self.assertTrue(parsed_flagged_zero.prefill_only)


if __name__ == "__main__":
    unittest.main()
