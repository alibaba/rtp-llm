#include "rtp_llm/cpp/model_rpc/K3StateTransfer.h"
#include <gtest/gtest.h>

using namespace rtp_llm;

TEST(K3StateTransferTest, StorageWireAndDestinationCompatibility) {
    for (auto storage : {TYPE_FP32, TYPE_BF16, TYPE_FP16}) {
        for (auto wire : {TYPE_INVALID, TYPE_FP32, TYPE_BF16, TYPE_FP16}) {
            for (auto destination : {TYPE_FP32, TYPE_BF16}) {
                GenerateRequestPB request;
                request.set_prefill_ssm_state_dtype(storage);
                if (wire != TYPE_INVALID) {
                    request.set_prefill_ssm_transfer_dtype(wire);
                }
                GenerateRequestPB decoded;
                ASSERT_TRUE(decoded.ParseFromString(request.SerializeAsString()));
                EXPECT_EQ((decoded.prefill_ssm_transfer_dtype_presence_case() == GenerateRequestPB::kPrefillSsmTransferDtype), wire != TYPE_INVALID);
                const bool expected = destination == TYPE_FP32
                    && (storage == TYPE_FP32 || storage == TYPE_BF16)
                    && (wire == TYPE_FP32 || (wire == TYPE_INVALID && storage == TYPE_FP32));
                EXPECT_EQ(supportsK3StateTransfer(decoded, destination), expected);
            }
        }
    }
}
