#include "rtp_llm/cpp/model_rpc/SamplerGeneratorState.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Functions.h>
#include <gtest/gtest.h>
#include <optional>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

namespace {

at::Generator makeGenerator(uint64_t seed) {
    auto generator = at::make_generator<at::CPUGeneratorImpl>();
    generator.set_current_seed(seed);
    return generator;
}

at::Tensor draw(at::Generator generator, int64_t count) {
    return at::rand({count}, std::optional<at::Generator>(generator), at::TensorOptions().dtype(at::kFloat));
}

}  // namespace

TEST(SamplerGeneratorStateTest, SerializedStateContinuesReferenceSequence) {
    auto source = makeGenerator(20260808);
    draw(source, 17);

    auto captured = captureSamplerGeneratorState(/*has_explicit_seed=*/true, source);
    ASSERT_TRUE(captured.ok()) << captured.status();

    GenerateRequestPB request;
    request.set_sampler_generator_state_version(kCurrentSamplerGeneratorStateVersion);
    request.set_sampler_generator_state(*captured);
    GenerateRequestPB parsed;
    ASSERT_TRUE(parsed.ParseFromString(request.SerializeAsString()));

    const auto expected = draw(source, 31);
    auto       restored = makeGenerator(1);
    const auto status   = restoreSamplerGeneratorState(
        parsed.sampler_generator_state_version(),
        /*has_explicit_seed=*/true,
        restored,
        parsed.sampler_generator_state());
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_TRUE(at::equal(draw(restored, 31), expected));
}

TEST(SamplerGeneratorStateTest, UnseededRequestDoesNotRequireState) {
    at::Generator undefined_generator;
    auto          captured = captureSamplerGeneratorState(/*has_explicit_seed=*/false, undefined_generator);
    ASSERT_TRUE(captured.ok()) << captured.status();
    EXPECT_TRUE(captured->empty());
    EXPECT_TRUE(restoreSamplerGeneratorState(
                    kCurrentSamplerGeneratorStateVersion,
                    /*has_explicit_seed=*/false,
                    undefined_generator,
                    /*serialized_state=*/"")
                    .ok());
}

TEST(SamplerGeneratorStateTest, LegacySeededRequestWithoutStateIsRejected) {
    auto legacy       = makeGenerator(7);
    auto initial_state = legacy.get_state().clone();

    const auto status = restoreSamplerGeneratorState(kLegacySamplerGeneratorStateVersion,
                                                     /*has_explicit_seed=*/true,
                                                     legacy,
                                                     /*serialized_state=*/"");
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "legacy seeded request cannot continue without sampler generator state");
    EXPECT_TRUE(at::equal(legacy.get_state(), initial_state));
}

TEST(SamplerGeneratorStateTest, LegacyUnseededRequestWithoutStateIsAccepted) {
    EXPECT_TRUE(restoreSamplerGeneratorState(kLegacySamplerGeneratorStateVersion,
                                             /*has_explicit_seed=*/false,
                                             at::Generator(),
                                             /*serialized_state=*/"")
                    .ok());
}

TEST(SamplerGeneratorStateTest, CurrentVersionSeededRequestRejectsMissingState) {
    const auto status = restoreSamplerGeneratorState(kCurrentSamplerGeneratorStateVersion,
                                                     /*has_explicit_seed=*/true,
                                                     makeGenerator(7),
                                                     /*serialized_state=*/"");
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(SamplerGeneratorStateTest, CurrentVersionSeededRequestRejectsTruncatedState) {
    auto source   = makeGenerator(7);
    auto captured = captureSamplerGeneratorState(/*has_explicit_seed=*/true, source);
    ASSERT_TRUE(captured.ok()) << captured.status();
    ASSERT_GT(captured->size(), 1);
    captured->pop_back();

    const auto status = restoreSamplerGeneratorState(
        kCurrentSamplerGeneratorStateVersion, /*has_explicit_seed=*/true, makeGenerator(7), *captured);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(SamplerGeneratorStateTest, CurrentVersionUnseededRequestRejectsUnexpectedState) {
    const auto status = restoreSamplerGeneratorState(
        kCurrentSamplerGeneratorStateVersion,
        /*has_explicit_seed=*/false,
        at::Generator(),
        /*serialized_state=*/"unexpected");
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(SamplerGeneratorStateTest, LegacyRequestRejectsUnexpectedState) {
    const auto status = restoreSamplerGeneratorState(kLegacySamplerGeneratorStateVersion,
                                                     /*has_explicit_seed=*/true,
                                                     makeGenerator(7),
                                                     /*serialized_state=*/"unexpected");
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(SamplerGeneratorStateTest, UnknownVersionIsRejected) {
    const auto status = restoreSamplerGeneratorState(/*wire_version=*/2,
                                                     /*has_explicit_seed=*/true,
                                                     makeGenerator(7),
                                                     /*serialized_state=*/"");
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace rtp_llm
