#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "torch/all.h"

#if USING_CUDA
#include <cuda_runtime.h>
#endif

#define private public
#define protected public
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#undef protected
#undef private

#include <xgrammar/tokenizer_info.h>

namespace rtp_llm {
namespace {

xgrammar::TokenizerInfo makeAsciiTokenizerInfo(int vocab_size = 128) {
    std::vector<std::string> vocab;
    vocab.reserve(static_cast<size_t>(vocab_size));
    for (int i = 0; i < vocab_size; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    return xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW, vocab_size, std::vector<int32_t>{0});
}

class ProbeModel: public ModelBase {
public:
    using ForwardHook       = std::function<void(const GptModelInputs&)>;
    using ReleaseBuffersHook = std::function<void()>;

    ProbeModel(size_t vocab_size, ForwardHook forward_hook = {}, ReleaseBuffersHook release_hook = {}):
        vocab_size_(vocab_size), forward_hook_(std::move(forward_hook)), release_hook_(std::move(release_hook)) {}

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        if (forward_hook_) {
            forward_hook_(inputs);
        }
        GptModelOutputs outputs;
        if (inputs.skip_lm_head) {
            return outputs;
        }
        const int64_t num_tokens = inputs.lm_output_indexes.defined() ? inputs.lm_output_indexes.size(0) : 1;
        // Favor lowercase 'b' (98) over out-of-grammar digits so real GrammarLogitsProcessor picks legal tokens.
        auto logits = torch::full({num_tokens, static_cast<int64_t>(vocab_size_)},
                                  -10.0f,
                                  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        logits.select(1, static_cast<int64_t>('0')).fill_(5.0f);
        logits.select(1, static_cast<int64_t>('b')).fill_(2.0f);
        outputs.logits = std::move(logits);
        return outputs;
    }

    void releaseBuffers() override {
        ++release_buffers_calls;
        if (release_hook_) {
            release_hook_();
        }
    }

    std::atomic<int> release_buffers_calls{0};

private:
    size_t             vocab_size_;
    ForwardHook        forward_hook_;
    ReleaseBuffersHook release_hook_;
};

class CountingStatefulProcessor: public BaseLogitsProcessor {
public:
    std::optional<ErrorInfo> process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override {
        (void)inputs;
        (void)start_idx;
        (void)finish_idx;
        ++process_calls;
        return std::nullopt;
    }

    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override {
        (void)src_batch_indices;
    }

    std::optional<ErrorInfo> updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override {
        (void)new_tokens;
        (void)num_new_tokens;
        ++update_calls;
        return std::nullopt;
    }

    bool isStateful() const override {
        return true;
    }

    bool supportsNormalAsyncDeviceState() const override {
        return false;
    }

    std::atomic<int> process_calls{0};
    std::atomic<int> update_calls{0};
};

class GatedBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    GatedBatchStreamProcessor(const ModelConfig&                  model_config,
                              const PDSepConfig&                  pd_sep_config,
                              const ProfilingDebugLoggingConfig&  profiling_debug_logging_config,
                              const CacheConfig&                  cache_config,
                              bool                                warm_up,
                              int                                 async_worker_count,
                              std::shared_ptr<std::promise<void>> dispatch_started,
                              std::shared_future<void>            release_dispatch):
        NormalBatchStreamProcessor(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, warm_up, async_worker_count),
        dispatch_started_(std::move(dispatch_started)),
        release_dispatch_(std::move(release_dispatch)) {}

    absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const override {
        // Model an in-flight async dispatch worker waiting on GPU D2H / event sync
        // before stream->update() acquires GenerateStream::mutex_.
        if (dispatch_calls.fetch_add(1, std::memory_order_acq_rel) == 0 && dispatch_started_) {
            dispatch_started_->set_value();
            release_dispatch_.wait();
        }
        return NormalBatchStreamProcessor::dispatch(stream_groups, merge_outputs);
    }

    mutable std::atomic<int>            dispatch_calls{0};
    std::shared_ptr<std::promise<void>> dispatch_started_;
    std::shared_future<void>            release_dispatch_;
};

class NormalExecutorCoAsyncTest: public DeviceTestBase {};

std::shared_ptr<NormalGenerateStream> makeDecodeStream(const ModelConfig& model_config,
                                                       const RuntimeConfig& runtime_config,
                                                       const ResourceContext& resource_context,
                                                       int32_t initial_token,
                                                       int     seq_len) {
    auto query                             = std::make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({initial_token}, torch::kInt32);
    query->generate_config                 = std::make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 16;
    query->generate_config->is_streaming   = true;
    query->generate_config->top_k          = 1;
    query->generate_config->do_sample      = false;

    auto stream                      = std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(false);
    stream->generate_status_->status = StreamState::RUNNING;
    const auto cuda_i32              = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
        .epoch                 = 0,
        .last_sample_token_gpu = torch::full({1}, static_cast<int64_t>(initial_token), cuda_i32),
        .next_seq_len_gpu      = torch::full({1}, static_cast<int64_t>(seq_len), cuda_i32),
        .last_real_seq_len     = seq_len - 1,
        .next_real_seq_len     = seq_len,
    });
    return stream;
}

TEST_F(NormalExecutorCoAsyncTest, DecodeOverlapsBlockingStatefulAndGrammarProcessorsUntilPreSamplerSync) {
    ASSERT_STREQ(std::getenv("RTP_LLM_STREAM_ASYNC"), "1");
    ASSERT_STREQ(std::getenv("RTP_LLM_DROP_BROAD_SYNC"), "1");
    ASSERT_STREQ(std::getenv("RTP_LLM_DEVICE_INPUT"), "1");

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    CustomConfig  custom;
    custom.output_dispatcher_worker_count = 1;
    auto params                           = createEngineInitParams(custom, model_config, runtime_config, kv_cache_config);
    params.model_config_.vocab_size       = 128;

    std::atomic<int>                    forward_count{0};
    std::shared_ptr<std::promise<void>> step2_forward_entered = std::make_shared<std::promise<void>>();
    auto                                step2_forward_future  = step2_forward_entered->get_future();

    NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
        return std::make_unique<ProbeModel>(128, [&](const GptModelInputs&) {
            if (++forward_count == 2) {
                step2_forward_entered->set_value();
            }
        });
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_guard;

    NormalExecutor executor(params, nullptr, /*warm_up=*/false);
    ASSERT_TRUE(executor.useStreamAsync());
    ASSERT_TRUE(executor.useDropBroadSync());
    ASSERT_TRUE(executor.useDeviceInput());

    GrammarConfig grammar_cfg;
    grammar_cfg.num_workers                  = 2;
    grammar_cfg.compiler_cache_bytes         = -1;
    grammar_cfg.terminate_without_stop_token = true;
    auto backend                             = XGrammarBackend::create(makeAsciiTokenizerInfo(128).SerializeJSON(), grammar_cfg);
    ASSERT_TRUE(backend != nullptr);
    auto matcher_or = backend->createMatcherFromKey({"regex", "[a-z]{4}"});
    ASSERT_TRUE(matcher_or.ok()) << matcher_or.status().ToString();
    auto grammar_proc = std::make_shared<GrammarLogitsProcessor>(matcher_or.value(), /*eos_token_id=*/0);

    auto dispatch_started = std::make_shared<std::promise<void>>();
    auto dispatch_entered = dispatch_started->get_future();
    std::promise<void> release_dispatch;
    auto               release_future = release_dispatch.get_future().share();
    bool               released       = false;
    struct ReleaseGuard {
        std::promise<void>& p;
        bool&               done;
        ~ReleaseGuard() {
            if (!done) {
                p.set_value();
            }
        }
    } release_guard{release_dispatch, released};

    executor.setBatchProcessor(std::make_unique<GatedBatchStreamProcessor>(params.model_config_,
                                                                           params.pd_sep_config,
                                                                           params.profiling_debug_logging_config,
                                                                           CacheConfig(),
                                                                           /*warm_up=*/false,
                                                                           custom.output_dispatcher_worker_count,
                                                                           dispatch_started,
                                                                           release_future));

    auto stateful_proc = std::make_shared<CountingStatefulProcessor>();

    ResourceContext resource_context;
    auto            stream = makeDecodeStream(params.model_config_, params.runtime_config, resource_context, 'a', 4);
    stream->sampling_state_.logits_processors.push_back(grammar_proc);
    stream->sampling_state_.logits_processors.push_back(stateful_proc);

    StreamGroups                 groups({stream});
    std::list<GenerateStreamPtr> stream_list{stream};

    // Preserve gatherCanUseDeviceState() contract when a blocking stateful processor is pending.
    stream->incPendingAsyncBookkeeping();
    EXPECT_FALSE(executor.gatherCanUseDeviceState(groups));
    stream->decPendingAsyncBookkeepingAndMaybeRelease();

    // Step 1 launches async dispatch worker, which pauses at dispatch() before stream->update().
    ASSERT_TRUE(executor.process(ScheduleOutput{stream_list}).ok());
    ASSERT_EQ(dispatch_entered.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_EQ(stateful_proc->process_calls.load(), 1);
    EXPECT_EQ(stateful_proc->update_calls.load(), 0);

    // Step 2 must enter model_->forward() without waiting for step 1 dispatch/updateStatus,
    // and must stall at wait_prev_dispatch_pre_sampler before running sampler_ / LogitsProcessor::process().
    auto step2_future = std::async(std::launch::async, [&]() {
        return executor.process(ScheduleOutput{stream_list});
    });
    ASSERT_EQ(step2_forward_future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(stateful_proc->process_calls.load(), 1);
    EXPECT_EQ(stateful_proc->update_calls.load(), 0);
    EXPECT_EQ(step2_future.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);

    released = true;
    release_dispatch.set_value();
    ASSERT_EQ(step2_future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    ASSERT_TRUE(step2_future.get().ok());

    // Step 3 confirms continuous sequential grammar masking ('b' beats masked '0') and committed state.
    ASSERT_TRUE(executor.process(ScheduleOutput{stream_list}).ok());
    executor.dispatch_runner_.sync(cuda_graph::graphGetCurrentStream());
    EXPECT_EQ(stateful_proc->process_calls.load(), 3);
    EXPECT_EQ(stateful_proc->update_calls.load(), 3);
    ASSERT_TRUE(grammar_proc->committedOutputLen().has_value());
    EXPECT_EQ(*grammar_proc->committedOutputLen(), 3);
}

#if USING_CUDA
struct StreamBlockState {
    std::shared_future<void> release;
    std::atomic<bool>        entered{false};
};

void CUDART_CB blockStreamCallback(void* data) {
    auto* state = reinterpret_cast<StreamBlockState*>(data);
    state->entered.store(true, std::memory_order_release);
    state->release.wait();
}

TEST_F(NormalExecutorCoAsyncTest, NonRootRankSkipsSyncAndDoubleReleaseOnlyOnAsyncNonWarmupDecode) {
    ASSERT_STREQ(std::getenv("RTP_LLM_STREAM_ASYNC"), "1");
    ASSERT_STREQ(std::getenv("RTP_LLM_DROP_BROAD_SYNC"), "1");
    ASSERT_STREQ(std::getenv("RTP_LLM_DEVICE_INPUT"), "1");

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params            = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    params.model_config_.vocab_size = 128;
    params.parallelism_config.tp_size = 1;
    params.parallelism_config.tp_rank = 1;

    for (const bool warm_up : {false, true}) {
        SCOPED_TRACE(testing::Message() << "warm_up=" << warm_up);
        ProbeModel* probe_ptr = nullptr;
        std::promise<void> release_gpu;
        StreamBlockState   block_state{release_gpu.get_future().share()};
        bool               released = false;
        struct GpuReleaseGuard {
            std::promise<void>& p;
            bool&               done;
            ~GpuReleaseGuard() {
                if (!done) {
                    p.set_value();
                }
            }
        } gpu_guard{release_gpu, released};

        std::atomic<bool> arm_stream_block{false};
        NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
            auto m    = std::make_unique<ProbeModel>(128, [&](const GptModelInputs&) {
                if (arm_stream_block.load(std::memory_order_acquire)) {
                    ASSERT_EQ(cudaLaunchHostFunc(
                                  cuda_graph::graphGetCurrentStream().stream(), blockStreamCallback, &block_state),
                              cudaSuccess);
                }
            });
            probe_ptr = m.get();
            return m;
        };
        struct FactoryResetGuard {
            ~FactoryResetGuard() {
                NormalExecutor::test_model_factory = nullptr;
            }
        } factory_guard;

        NormalExecutor executor(params, nullptr, warm_up);
        ASSERT_NE(probe_ptr, nullptr);

        ResourceContext resource_context;
        auto            decode_stream = makeDecodeStream(params.model_config_, params.runtime_config, resource_context, 'a', 4);
        std::list<GenerateStreamPtr> decode_list{decode_stream};

        probe_ptr->release_buffers_calls.store(0);
        arm_stream_block.store(true, std::memory_order_release);
        auto proc_future = std::async(std::launch::async, [&]() {
            return executor.process(ScheduleOutput{decode_list});
        });

        if (!warm_up) {
            ASSERT_EQ(proc_future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
            ASSERT_TRUE(proc_future.get().ok());
            // Only the per-step L380 release runs; L440 must not release a second round prematurely.
            EXPECT_EQ(probe_ptr->release_buffers_calls.load(), 1);
            released = true;
            release_gpu.set_value();
            cudaSyncAndCheck();
        } else {
            while (!block_state.entered.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            EXPECT_EQ(proc_future.wait_for(std::chrono::milliseconds(40)), std::future_status::timeout);
            released = true;
            release_gpu.set_value();
            ASSERT_EQ(proc_future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
            ASSERT_TRUE(proc_future.get().ok());
            // L380 + L440 both run when cudaSyncAndCheck() drained the step.
            EXPECT_EQ(probe_ptr->release_buffers_calls.load(), 2);
        }
    }

    // Prefill on non-warmup tp_rank_ > 0 must still execute cudaSyncAndCheck() and releaseBuffers().
    ProbeModel*        probe_ptr = nullptr;
    std::promise<void> release_prefill_gpu;
    StreamBlockState   prefill_block_state{release_prefill_gpu.get_future().share()};
    bool               prefill_released = false;
    struct PrefillReleaseGuard {
        std::promise<void>& p;
        bool&               done;
        ~PrefillReleaseGuard() {
            if (!done) {
                p.set_value();
            }
        }
    } prefill_guard{release_prefill_gpu, prefill_released};

    NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
        auto m    = std::make_unique<ProbeModel>(128, [&](const GptModelInputs&) {
            ASSERT_EQ(cudaLaunchHostFunc(cuda_graph::graphGetCurrentStream().stream(),
                                         blockStreamCallback,
                                         &prefill_block_state),
                      cudaSuccess);
        });
        probe_ptr = m.get();
        return m;
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_guard;

    NormalExecutor  executor(params, nullptr, /*warm_up=*/false);
    ResourceContext resource_context;
    auto query                             = std::make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2, 3}, torch::kInt32);
    query->generate_config                 = std::make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 4;
    auto prefill_stream =
        std::make_shared<NormalGenerateStream>(query, params.model_config_, params.runtime_config, resource_context, nullptr);
    prefill_stream->setIsContextStream(true);
    prefill_stream->generate_status_->status = StreamState::RUNNING;

    auto prefill_future = std::async(std::launch::async, [&]() {
        return executor.process(ScheduleOutput{{prefill_stream}});
    });
    while (!prefill_block_state.entered.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(prefill_future.wait_for(std::chrono::milliseconds(40)), std::future_status::timeout);
    prefill_released = true;
    release_prefill_gpu.set_value();
    ASSERT_EQ(prefill_future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    ASSERT_TRUE(prefill_future.get().ok());
    EXPECT_EQ(probe_ptr->release_buffers_calls.load(), 2);
}

TEST_F(NormalExecutorCoAsyncTest, MultiStreamPublishNormalDeviceStateBatchesUpdatesAndPreservesPriorSeqLen) {
    ASSERT_STREQ(std::getenv("RTP_LLM_DEVICE_INPUT"), "1");

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params            = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    params.model_config_.vocab_size = 128;

    NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
        return std::make_unique<ProbeModel>(128);
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_guard;

    NormalExecutor executor(params, nullptr, /*warm_up=*/false);
    ResourceContext resource_context;
    const auto      cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    auto s0 = makeDecodeStream(params.model_config_, params.runtime_config, resource_context, 'a', 4);
    auto s1 = makeDecodeStream(params.model_config_, params.runtime_config, resource_context, 'b', 7);
    auto s2 = makeDecodeStream(params.model_config_, params.runtime_config, resource_context, 'c', 10);
    // Exercise mixed missing initial device state alongside populated decode streams.
    s1->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{});

    const auto prev_s0_seq = s0->getNormalAsyncDeviceState().next_seq_len_gpu;
    const auto prev_s2_seq = s2->getNormalAsyncDeviceState().next_seq_len_gpu;

    std::list<GenerateStreamPtr> stream_list{s0, s1, s2};
    StreamGroups                 groups(stream_list);
    SamplerOutput                sampler_output;
    sampler_output.token_ids = torch::tensor({11, 22, 33}, cuda_i32);

    executor.publishNormalDeviceState(groups, sampler_output);
    cudaSyncAndCheck();

    // Prior next_seq_len_gpu tensors must remain immutable for concurrent readers.
    EXPECT_EQ(prev_s0_seq.cpu().item<int32_t>(), 4);
    EXPECT_EQ(prev_s2_seq.cpu().item<int32_t>(), 10);

    EXPECT_EQ(s0->getNormalAsyncDeviceState().last_sample_token_gpu.cpu().item<int32_t>(), 11);
    EXPECT_EQ(s0->getNormalAsyncDeviceState().next_seq_len_gpu.cpu().item<int32_t>(), 5);
    EXPECT_EQ(s0->getNormalAsyncDeviceState().next_real_seq_len, 5);

    EXPECT_EQ(s1->getNormalAsyncDeviceState().last_sample_token_gpu.cpu().item<int32_t>(), 22);
    // s1 had no prior state and stream->seqLength() == 1 (1 initial token), so next == 2.
    EXPECT_EQ(s1->getNormalAsyncDeviceState().next_seq_len_gpu.cpu().item<int32_t>(), 2);
    EXPECT_EQ(s1->getNormalAsyncDeviceState().next_real_seq_len, 2);

    EXPECT_EQ(s2->getNormalAsyncDeviceState().last_sample_token_gpu.cpu().item<int32_t>(), 33);
    EXPECT_EQ(s2->getNormalAsyncDeviceState().next_seq_len_gpu.cpu().item<int32_t>(), 11);
    EXPECT_EQ(s2->getNormalAsyncDeviceState().next_real_seq_len, 11);
}
#endif

}  // namespace
}  // namespace rtp_llm
