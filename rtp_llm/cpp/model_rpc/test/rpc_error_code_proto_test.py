import unittest

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.descriptor import FieldDescriptor
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (  # type: ignore
    DESCRIPTOR,
    BatchGenerateInputPB,
    BatchGenerateOutputsPB,
    ErrorCodePB,
    ErrorDetailsPB,
    GenerateConfigPB,
    RpcErrorPB,
)


class RpcErrorCodeProtoTest(unittest.TestCase):
    def test_capture_has_independent_default_false_wire_field(self):
        field = GenerateConfigPB.DESCRIPTOR.fields_by_name["capture_hidden_states"]
        self.assertEqual(field.number, 77)
        self.assertEqual(field.type, FieldDescriptor.TYPE_BOOL)
        self.assertFalse(GenerateConfigPB.FromString(b"").capture_hidden_states)
        for capture in (False, True):
            wire = GenerateConfigPB(
                max_new_tokens=4, reuse_cache=True, capture_hidden_states=capture
            )
            parsed = GenerateConfigPB.FromString(wire.SerializeToString())
            self.assertEqual(parsed.capture_hidden_states, capture)
            self.assertTrue(parsed.reuse_cache)
            self.assertEqual(parsed.max_new_tokens, 4)
            self.assertFalse(parsed.prefill_only)

    def test_rpc_error_codes_are_declared_and_round_trip(self):
        for name, number, message in (
            ("INVALID_PARAMS", 26, "invalid zero-prefill config"),
            ("EXECUTION_EXCEPTION", 27, "handler threw"),
            ("HIDDEN_STATE_CAPTURE_BACKPRESSURE", 28, ""),
            ("HIDDEN_STATE_CAPTURE_BACKPRESSURE", 28, "capture quota occupied"),
        ):
            with self.subTest(name=name):
                error_code = ErrorCodePB.Value(name)
                self.assertEqual(error_code, number)

                serialized = RpcErrorPB(
                    error_code=error_code,
                    error_message=message,
                ).SerializeToString()
                parsed = RpcErrorPB.FromString(serialized)

                self.assertEqual(parsed.error_code, error_code)
                self.assertEqual(ErrorCodePB.Name(parsed.error_code), name)
                self.assertEqual(parsed.error_message, message)

    def test_native_details_and_batch_rpc_use_independent_error_number_spaces(self):
        details = ErrorDetailsPB(
            error_code=8107, error_message="capture quota occupied"
        )
        parsed = ErrorDetailsPB.FromString(details.SerializeToString())
        self.assertEqual(parsed.error_code, 8107)
        response = BatchGenerateOutputsPB()
        response.results.add().final_output.request_id = 10
        rejected = response.results.add().error_info
        rejected.error_code = ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE
        rejected.error_message = details.error_message
        response.results.add().final_output.request_id = 12
        parsed = BatchGenerateOutputsPB.FromString(response.SerializeToString())
        self.assertEqual(len(parsed.results), 3)
        self.assertEqual(parsed.results[0].final_output.request_id, 10)
        self.assertEqual(parsed.results[1].error_info.error_code, 28)
        self.assertEqual(parsed.results[2].final_output.request_id, 12)

    def test_batch_native_details_field_and_legacy_wire_compatibility(self):
        fields = BatchGenerateOutputsPB().results.add().DESCRIPTOR.fields_by_name
        self.assertEqual(fields["final_output"].number, 1)
        self.assertEqual(fields["error_info"].number, 2)
        self.assertEqual(fields["error_details"].number, 3)
        self.assertEqual(fields["error_details"].message_type.full_name, ErrorDetailsPB.DESCRIPTOR.full_name)
        self.assertEqual(ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE, 28)
        for code in (603, 511, 607, 602, 8211, 8212, 8213, 99999, 8107):
            with self.subTest(code=code):
                response = BatchGenerateOutputsPB()
                response.results.add().final_output.request_id = 100
                failure = response.results.add()
                failure.error_info.error_code = 28 if code == 8107 else 1
                failure.error_info.error_message = "native reason"
                failure.error_details.error_code = code
                failure.error_details.error_message = "native reason"
                response.results.add().final_output.request_id = 102
                wire = response.SerializeToString()
                parsed = BatchGenerateOutputsPB.FromString(wire)
                self.assertEqual(len(parsed.results), 3)
                self.assertEqual(parsed.results[0].final_output.request_id, 100)
                self.assertFalse(parsed.results[0].HasField("error_details"))
                self.assertEqual(parsed.results[1].error_details.error_code, code)
                self.assertEqual(parsed.results[1].error_details.error_message, "native reason")
                self.assertEqual(parsed.results[1].error_info, failure.error_info)
                self.assertEqual(parsed.results[2].final_output.request_id, 102)

                # Parse the same bytes with a real pre-field-3 descriptor, not a text assertion.
                file_proto = descriptor_pb2.FileDescriptorProto()
                DESCRIPTOR.CopyToProto(file_proto)
                batch_result = next(m for m in file_proto.message_type if m.name == "BatchGenerateResultPB")
                native_index = next(i for i, field in enumerate(batch_result.field) if field.number == 3)
                del batch_result.field[native_index]
                pool = descriptor_pool.DescriptorPool()
                for dependency in DESCRIPTOR.dependencies:
                    pool.AddSerializedFile(dependency.serialized_pb)
                pool.Add(file_proto)
                legacy_type = message_factory.GetMessageClass(pool.FindMessageTypeByName("BatchGenerateOutputsPB"))
                legacy = legacy_type.FromString(wire)
                self.assertEqual(legacy.results[1].error_info.error_code, failure.error_info.error_code)
                self.assertEqual(legacy.results[1].error_info.error_message, "native reason")
                self.assertEqual(legacy.results[2].final_output.request_id, 102)
                legacy.DiscardUnknownFields()
                fallback = BatchGenerateOutputsPB.FromString(legacy.SerializeToString())
                self.assertFalse(fallback.results[1].HasField("error_details"))
                self.assertEqual(fallback.results[1].error_info, failure.error_info)

    def test_native_batch_service_and_input_wire_contract(self):
        method = DESCRIPTOR.services_by_name["RpcService"].methods_by_name[
            "BatchGenerateCall"
        ]
        self.assertEqual(method.input_type, BatchGenerateInputPB.DESCRIPTOR)
        self.assertEqual(method.output_type, BatchGenerateOutputsPB.DESCRIPTOR)
        self.assertFalse(method.client_streaming)
        self.assertFalse(method.server_streaming)
        batch = BatchGenerateInputPB()
        for request_id in (10, 11, 12):
            item = batch.inputs.add(request_id=request_id)
            item.generate_config.CopyFrom(
                GenerateConfigPB(
                    max_new_tokens=0,
                    prefill_only=True,
                    capture_hidden_states=True,
                    reuse_cache=False,
                )
            )
        parsed = BatchGenerateInputPB.FromString(batch.SerializeToString())
        self.assertEqual([item.request_id for item in parsed.inputs], [10, 11, 12])
        self.assertTrue(
            all(item.generate_config.capture_hidden_states for item in parsed.inputs)
        )

    def test_base_logprobs_and_prefill_only_keep_independent_wire_fields(self):
        fields = GenerateConfigPB.DESCRIPTOR.fields_by_name
        self.assertEqual(fields["return_logprobs"].number, 64)
        self.assertEqual(fields["top_logprobs"].number, 65)
        self.assertEqual(fields["frontend_metric_streaming"].number, 66)
        self.assertEqual(fields["prefill_only"].number, 76)
        for prefill_only in (False, True):
            config = GenerateConfigPB(
                max_new_tokens=0,
                prefill_only=prefill_only,
                return_logprobs=True,
                top_logprobs=2,
                frontend_metric_streaming=True,
            )
            parsed = GenerateConfigPB.FromString(config.SerializeToString())
            self.assertEqual(parsed.prefill_only, prefill_only)
            self.assertTrue(parsed.return_logprobs)
            self.assertEqual(parsed.top_logprobs, 2)
            self.assertTrue(parsed.frontend_metric_streaming)

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
