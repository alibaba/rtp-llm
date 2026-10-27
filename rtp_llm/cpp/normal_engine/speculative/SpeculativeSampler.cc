#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <map>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <limits>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA
#include <ATen/cuda/CUDAGraph.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect_fp32.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#endif

namespace rtp_llm {
namespace speculative {

int dsparkForceTopK() {
    const char* value = std::getenv("RTP_LLM_DSPARK_FORCE_TOPK");
    if (value == nullptr) {
        return 4096;
    }
    char* end = nullptr;
    errno     = 0;
    long topk = std::strtol(value, &end, 10);
    RTP_LLM_CHECK_WITH_INFO(errno == 0 && end != value && *end == '\0' && topk >= 0
                                && topk <= std::numeric_limits<int>::max(),
                            "RTP_LLM_DSPARK_FORCE_TOPK must be an integer in [0, INT_MAX], got '%s'",
                            value);
    return static_cast<int>(topk);
}

#if USING_CUDA
struct DSparkSamplingGraphs {
    struct Entry {
        at::cuda::CUDAGraph graph;
        torch::Tensor       anchors;
        torch::Tensor       temperatures;
        SamplerOutput       output;
    };
    torch::Tensor          logits, w1, w2, map;
    const void *           logits_ptr = nullptr, *w1_ptr = nullptr, *w2_ptr = nullptr, *map_ptr = nullptr;
    size_t                 vocab;
    c10::cuda::CUDAStream  capture_stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStream  replay_stream  = at::cuda::getCurrentCUDAStream();
    c10::cuda::MempoolId_t pool           = at::cuda::graph_pool_handle();
    std::unordered_map<int64_t, std::unique_ptr<Entry>> entries;

    ~DSparkSamplingGraphs() {
        try {
            const c10::cuda::CUDAGuard guard(logits.device());
            replay_stream.synchronize();
            capture_stream.synchronize();
        } catch (const std::exception& error) {
            RTP_LLM_LOG_WARNING("DSpARK sampling graph drain failed: %s", error.what());
        }
    }
};

struct RejectionSamplingGraphs {
    struct Entry {
        cudaGraph_t                   graph      = nullptr;
        cudaGraphExec_t               executable = nullptr;
        cudaGraphNode_t               kernel     = nullptr;
        cudaKernelNodeParams          kernel_params{};
        torch::Tensor                 masks_h, masks_d, draft, target, uniform, result_d, result_h;
        std::shared_ptr<torch::Event> ready    = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        torch::Event                  uploaded = cuda_graph::makeGraphEvent(), computed = cuda_graph::makeGraphEvent();
        bool                          submitted = false;

        ~Entry() {
            if (!executable && !graph)
                return;
            const c10::cuda::CUDAGuard guard(masks_d.device());
            if (executable)
                cudaGraphExecDestroy(executable);
            if (graph)
                cudaGraphDestroy(graph);
        }
    };
    torch::Device                                          device;
    int                                                    vocab;
    c10::cuda::CUDAStream                                  copy_stream = at::cuda::getStreamFromPool();
    std::map<int64_t, std::vector<std::shared_ptr<Entry>>> entries;

    explicit RejectionSamplingGraphs(torch::Device d, int v): device(d), vocab(v) {}
    ~RejectionSamplingGraphs() {
        try {
            const c10::cuda::CUDAGuard guard(device);
            for (const auto& [batch, slots] : entries)
                for (const auto& slot : slots)
                    if (slot->submitted)
                        slot->ready->synchronize();
        } catch (const std::exception& error) {
            RTP_LLM_LOG_WARNING("rejection graph drain failed: %s", error.what());
        }
    }
};

static bool sameGraphTensor(const torch::Tensor& a, const torch::Tensor& b, const void* captured_ptr) {
    if (!a.defined() || !b.defined())
        return a.defined() == b.defined();
    return a.device() == b.device() && a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes()
           && a.strides() == b.strides() && a.data_ptr() == captured_ptr;
}
#endif

void SpeculativeSampler::prepareRejectionGraphs(const torch::Device&    device,
                                                int                     vocab,
                                                const std::vector<int>& batch_sizes) {
#if USING_CUDA
    if (!device.is_cuda() || vocab <= 0 || propose_step_ == 0 || batch_sizes.empty())
        return;
    const c10::cuda::CUDAGuard guard(device);
    auto                       graphs = std::make_shared<RejectionSamplingGraphs>(device, vocab);
    const auto                 stream = graphs->copy_stream;
    cuda_graph::graphDeviceSynchronize();
    const c10::cuda::CUDAStreamGuard stream_guard(stream);
    const auto                       ints = torch::TensorOptions().device(device).dtype(torch::kInt32);
    const auto                       host = torch::TensorOptions().device(torch::kCPU).pinned_memory(true);
    for (int batch : batch_sizes) {
        if (batch <= 0 || graphs->entries.count(batch))
            continue;
        auto& slots = graphs->entries[batch];
        for (int i = 0; i < 3; ++i) {
            auto e      = std::make_shared<RejectionSamplingGraphs::Entry>();
            e->masks_h  = torch::zeros({2, batch}, host.dtype(torch::kBool));
            e->masks_d  = torch::empty({2, batch}, ints.dtype(torch::kBool));
            e->draft    = torch::zeros({batch, static_cast<int64_t>(propose_step_)}, ints);
            e->target   = torch::zeros({batch * static_cast<int64_t>(propose_step_ + 1), 1}, ints);
            e->uniform  = torch::zeros({batch, static_cast<int64_t>(propose_step_ + 1)}, ints.dtype(torch::kFloat32));
            e->result_d = torch::empty({batch * static_cast<int64_t>(propose_step_ + 2)}, ints);
            e->result_h = torch::empty(e->result_d.sizes(), host.dtype(torch::kInt32));
            auto run    = [&]() {
                // Both probability distributions are point masses. Keep the
                // original rejection kernel, with its exact acceptance and
                // force-override rules; dense/mixed sampling stays eager.
                const auto status = invokeRejectionSampling<float, int32_t>(nullptr,
                                                                            e->draft.data_ptr<int32_t>(),
                                                                            e->uniform.data_ptr<float>(),
                                                                            nullptr,
                                                                            e->target.data_ptr<int32_t>(),
                                                                            1,
                                                                            e->result_d.data_ptr<int32_t>(),
                                                                            e->result_d.data_ptr<int32_t>()
                                                                                + batch * (propose_step_ + 1),
                                                                            e->masks_d.data_ptr<bool>(),
                                                                            batch,
                                                                            propose_step_,
                                                                            vocab,
                                                                            stream.stream(),
                                                                            true,
                                                                            true,
                                                                            e->masks_d.data_ptr<bool>() + batch,
                                                                            true);
                C10_CUDA_CHECK(status);
            };
            C10_CUDA_CHECK(cudaMemcpyAsync(e->masks_d.data_ptr(),
                                           e->masks_h.data_ptr(),
                                           e->masks_d.nbytes(),
                                           cudaMemcpyHostToDevice,
                                           stream.stream()));
            run();
            stream.synchronize();
            // No allocation or RNG inside this capture. Transfers use an
            // independent stream at runtime, so DMA cannot stall model compute.
            C10_CUDA_CHECK(cudaStreamBeginCapture(stream.stream(), cudaStreamCaptureModeGlobal));
            try {
                run();
            } catch (...) {
                cudaStreamEndCapture(stream.stream(), &e->graph);
                throw;
            }
            C10_CUDA_CHECK(cudaStreamEndCapture(stream.stream(), &e->graph));
            size_t count = 0;
            C10_CUDA_CHECK(cudaGraphGetNodes(e->graph, nullptr, &count));
            std::vector<cudaGraphNode_t> nodes(count);
            C10_CUDA_CHECK(cudaGraphGetNodes(e->graph, nodes.data(), &count));
            for (auto node : nodes) {
                cudaGraphNodeType type;
                C10_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
                if (type == cudaGraphNodeTypeKernel) {
                    e->kernel = node;
                    C10_CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &e->kernel_params));
                }
            }
            RTP_LLM_CHECK_WITH_INFO(e->kernel, "incomplete rejection sampling graph");
            C10_CUDA_CHECK(cudaGraphInstantiateWithFlags(&e->executable, e->graph, 0));
            C10_CUDA_CHECK(cudaGraphLaunch(e->executable, stream.stream()));
            stream.synchronize();
            slots.push_back(std::move(e));
        }
    }
    rejection_graphs_ = std::move(graphs);
#endif
}

bool SpeculativeSampler::replayRejectionGraph(SpeculativeSamplerOutput& output,
                                              const torch::Tensor&      masks,
                                              const torch::Tensor&      draft,
                                              const torch::Tensor&      target,
                                              const torch::Tensor&      uniform,
                                              const torch::Tensor&      probs,
                                              bool                      point_mass,
                                              const torch::Tensor&      valid_draft_lengths) const {
#if USING_CUDA
    const auto& graphs = rejection_graphs_;
    if (!graphs || !point_mass || draft.device() != graphs->device || target.device() != graphs->device
        || uniform.device() != graphs->device || draft.scalar_type() != torch::kInt32
        || target.scalar_type() != torch::kInt32 || !draft.is_contiguous() || !target.is_contiguous()
        || draft.dim() != 2 || draft.size(0) == 0 || draft.size(1) != static_cast<int64_t>(propose_step_)
        || target.dim() != 2 || target.size(1) < 1 || target.size(0) != draft.size(0) * (propose_step_ + 1)
        || probs.dim() != 3 || probs.size(2) != graphs->vocab)
        return false;
    const int64_t batch = draft.size(0), token_count = batch * (propose_step_ + 1);
    const auto    found = graphs->entries.lower_bound(batch);
    if (found == graphs->entries.end())
        return false;
    const c10::cuda::CUDAGuard guard(graphs->device);
    for (const auto& entry : found->second) {
        // Host result ownership and upload completion are separate: a done
        // event alone does not mean asynchronous CPU bookkeeping has finished.
        if (entry.use_count() != 1 || entry->result_h.storage().use_count() != 1
            || (entry->submitted && !entry->ready->query()))
            continue;
        entry->masks_h.zero_();
        entry->masks_h.narrow(1, 0, batch).copy_(masks);
        const auto main_stream = at::cuda::getCurrentCUDAStream();
        auto       copy_stream = graphs->copy_stream;
        while (copy_stream == main_stream)
            copy_stream = at::cuda::getStreamFromPool();
        auto result = torch::empty({batch * static_cast<int64_t>(propose_step_ + 2)}, entry->result_d.options());
        C10_CUDA_CHECK(cudaMemcpyAsync(entry->masks_d.data_ptr(),
                                       entry->masks_h.data_ptr(),
                                       entry->masks_d.nbytes(),
                                       cudaMemcpyHostToDevice,
                                       copy_stream.stream()));
        entry->uploaded.record(copy_stream);
        entry->uploaded.block(main_stream);
        // Patch node arguments to this call's tensors. The original kernel
        // reads the last history column directly; no staging or output clone
        // is needed, and every GPU result owns independent storage.
        float*         draft_probs    = nullptr;
        float*         target_probs   = nullptr;
        auto*          draft_ids      = draft.data_ptr<int32_t>();
        auto*          target_ids     = target.data_ptr<int32_t>();
        auto*          samples        = uniform.data_ptr<float>();
        auto*          output_ids     = result.data_ptr<int32_t>();
        auto*          output_lengths = output_ids + token_count;
        auto*          do_sample      = entry->masks_d.data_ptr<bool>();
        auto*          force          = do_sample + found->first;
        int            rows = batch, steps = propose_step_, vocab = graphs->vocab, stride = target.size(1);
        bool           draft_point_mass = true, target_point_mass = true, finalize = true;
        const int32_t* valid_lengths =
            valid_draft_lengths.defined() ? valid_draft_lengths.data_ptr<int32_t>() : nullptr;
        void* args[]        = {&draft_probs,
                               &draft_ids,
                               &samples,
                               &target_probs,
                               &target_ids,
                               &stride,
                               &output_ids,
                               &output_lengths,
                               &do_sample,
                               &rows,
                               &steps,
                               &vocab,
                               &draft_point_mass,
                               &target_point_mass,
                               &force,
                               &finalize,
                               &valid_lengths};
        auto  params        = entry->kernel_params;
        params.gridDim.x    = rows;
        params.kernelParams = args;
        C10_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(entry->executable, entry->kernel, &params));
        C10_CUDA_CHECK(cudaGraphLaunch(entry->executable, main_stream.stream()));
        entry->computed.record(main_stream);
        entry->computed.block(copy_stream);
        C10_CUDA_CHECK(cudaMemcpyAsync(entry->result_h.data_ptr(),
                                       result.data_ptr(),
                                       result.nbytes(),
                                       cudaMemcpyDeviceToHost,
                                       copy_stream.stream()));
        c10::cuda::CUDACachingAllocator::recordStream(result.storage().data_ptr(), copy_stream);
        output.accept_tokens = result.narrow(0, 0, token_count).view({batch, static_cast<int64_t>(propose_step_ + 1)});
        output.accept_len    = result.narrow(0, token_count, batch);
        output.accept_tokens_cpu = entry->result_h.narrow(0, 0, token_count).view(output.accept_tokens.sizes());
        output.accept_len_cpu    = entry->result_h.narrow(0, token_count, batch);
        entry->ready->record(copy_stream);
        entry->submitted = true;
        // A later logits cap may re-record the public completion event.
        // Slot reuse must continue to track this original transfer separately.
        output.transfer_done_event->record(copy_stream);
        output.graph_buffers = entry;
        return true;
    }
#endif
    return false;
}

void SpeculativeSampler::prepareDSparkDraftGraphs(const torch::Tensor&    logits_buffer,
                                                  const torch::Tensor&    markov_w1,
                                                  const torch::Tensor&    markov_w2,
                                                  size_t                  draft_vocab_size,
                                                  const std::vector<int>& batch_sizes) {
#if USING_CUDA
    if (!logits_buffer.defined() || !logits_buffer.is_cuda() || logits_buffer.dim() != 2
        || logits_buffer.scalar_type() != torch::kFloat32 || !logits_buffer.is_contiguous() || propose_step_ == 0
        || draft_vocab_size >= (1u << 24))
        return;
    const c10::cuda::CUDAGuard device_guard(logits_buffer.device());
    auto                       graphs = std::make_shared<DSparkSamplingGraphs>();
    graphs->logits                    = logits_buffer.detach();
    graphs->w1                        = markov_w1.detach();
    graphs->w2                        = markov_w2.detach();
    graphs->map                       = d2t_map_.defined() ? d2t_map_.detach() : torch::Tensor();
    graphs->logits_ptr                = logits_buffer.data_ptr();
    graphs->w1_ptr                    = markov_w1.data_ptr();
    graphs->w2_ptr                    = markov_w2.data_ptr();
    graphs->map_ptr                   = d2t_map_.defined() ? d2t_map_.data_ptr() : nullptr;
    graphs->vocab                     = draft_vocab_size;
    auto sizes                        = batch_sizes;
    if (sizes.empty())
        sizes.push_back(logits_buffer.size(0) / propose_step_);
    std::sort(sizes.begin(), sizes.end(), std::greater<int>());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    cuda_graph::graphDeviceSynchronize();
    const c10::cuda::CUDAStreamGuard stream_guard(graphs->capture_stream);
    // Keep every owning graph alive while its private pool is shared.
    dspark_graphs_ = graphs;
    for (int batch : sizes) {
        if (batch <= 0 || static_cast<int64_t>(batch) * propose_step_ > logits_buffer.size(0))
            continue;
        auto entry          = std::make_unique<DSparkSamplingGraphs::Entry>();
        entry->anchors      = torch::zeros({batch}, logits_buffer.options().dtype(torch::kInt32));
        entry->temperatures = torch::full({batch}, 1.0e-6f, logits_buffer.options());
        auto input          = logits_buffer.narrow(0, 0, batch * propose_step_);
        auto run            = [&]() {
            return sampleDSparkDraftEager(
                input, entry->anchors, entry->temperatures, {}, {}, markov_w1, markov_w2, draft_vocab_size, true, 0, 0);
        };
        run();
        run();
        graphs->capture_stream.synchronize();
        entry->graph.capture_begin(graphs->pool);
        entry->output = run();
        entry->graph.capture_end();
        entry->graph.replay();
        graphs->capture_stream.synchronize();
        graphs->entries.emplace(batch, std::move(entry));
    }
#endif
}

FastTopKSamplerOutput FastTopKSampler::forward(const torch::Tensor& logits, int top_k) {
    FastTopKSamplerOutput output;

    if (top_k == 1) {
        output.token_ids = torch::argmax(logits, -1, true);
        output.all_probs = torch::zeros_like(logits).scatter_(-1, output.token_ids, 1.0);
    } else {
        auto draft_probs = torch::softmax(logits, -1);
        output.token_ids = std::get<1>(torch::topk(draft_probs, top_k, -1));
        output.all_probs = std::move(draft_probs);
    }

    int batch_size = output.token_ids.size(0);
    execMappingDraft2Target({output.token_ids, d2t_map_, batch_size, 0, 1});

    return output;
}

torch::Tensor SpeculativeSampler::buildValidDraftLengths(const torch::Tensor& confidence_rates,
                                                         const torch::Tensor& stable_order_indices,
                                                         int64_t              total_draft_budget) {
    RTP_LLM_CHECK_WITH_INFO(confidence_rates.defined() && confidence_rates.dim() == 2
                                && confidence_rates.scalar_type() == torch::kFloat32 && confidence_rates.is_contiguous()
                                && (confidence_rates.device().is_cpu() || confidence_rates.is_cuda()),
                            "DSpARK prefix planner requires contiguous CPU/CUDA FP32 [B, gamma] rates");
    const int64_t batch_size = confidence_rates.size(0);
    const int64_t gamma      = confidence_rates.size(1);
    RTP_LLM_CHECK_WITH_INFO(gamma > 0 && gamma <= std::numeric_limits<int32_t>::max(),
                            "DSpARK prefix planner requires a positive int32 proposal width");
    RTP_LLM_CHECK_WITH_INFO(
        stable_order_indices.defined() && stable_order_indices.dim() == 1 && stable_order_indices.size(0) == batch_size
            && stable_order_indices.scalar_type() == torch::kInt64 && stable_order_indices.is_contiguous()
            && stable_order_indices.device() == confidence_rates.device(),
        "DSpARK prefix planner requires contiguous int64 [B] stable row indices on the rates device");
    RTP_LLM_CHECK_WITH_INFO(total_draft_budget >= 0 && total_draft_budget <= confidence_rates.numel(),
                            "DSpARK prefix planner budget must be between zero and B*gamma");
    const auto length_options = confidence_rates.options().dtype(torch::kInt32);
    if (batch_size == 0 || total_draft_budget == 0) {
        return torch::zeros({batch_size}, length_options);
    }
    if (total_draft_budget == confidence_rates.numel()) {
        return torch::full({batch_size}, gamma, length_options);
    }

    auto valid =
        torch::isfinite(confidence_rates).logical_and(confidence_rates.ge(0.0)).logical_and(confidence_rates.le(1.0));
    auto clean = torch::where(valid, confidence_rates, torch::zeros_like(confidence_rates));
    // Parallel cumprod may use different floating-point association at adjacent
    // positions. A prefix minimum prevents a rounded successor from overtaking
    // its predecessor. It reads only earlier scores, preserving causality.
    auto scores = std::get<0>(clean.cumprod(1).cummin(1)).index_select(0, stable_order_indices);
    // Request-major flattening already orders ties by stable identity and then
    // position. A stable score sort preserves that order without epsilon keys.
    auto order    = torch::argsort(scores.reshape({-1}), /*stable=*/true, /*dim=*/0, /*descending=*/true);
    auto selected = torch::zeros({confidence_rates.numel()}, length_options);
    selected.index_fill_(0, order.narrow(0, 0, total_draft_budget), 1);
    auto sorted_lengths = selected.view({batch_size, gamma}).sum(1, false, torch::kInt32);
    auto lengths        = torch::empty({batch_size}, length_options);
    lengths.index_copy_(0, stable_order_indices, sorted_lengths);
    return lengths;
}

torch::Tensor SpeculativeSampler::buildConfidenceStats(const torch::Tensor& confidence_rates,
                                                       const torch::Tensor& accept_len,
                                                       const torch::Tensor& valid_lengths,
                                                       const torch::Tensor& eligible_rows) {
    RTP_LLM_CHECK_WITH_INFO(confidence_rates.defined() && confidence_rates.dim() == 2
                                && confidence_rates.scalar_type() == torch::kFloat32,
                            "DSpARK confidence statistics require FP32 [B, gamma] rates");
    const auto batch_size          = confidence_rates.size(0);
    const auto gamma               = confidence_rates.size(1);
    const auto valid_length_tensor = [&](const torch::Tensor& tensor) {
        return tensor.defined() && tensor.dim() == 1 && tensor.size(0) == batch_size
               && tensor.device() == confidence_rates.device()
               && (tensor.scalar_type() == torch::kInt32 || tensor.scalar_type() == torch::kInt64);
    };
    RTP_LLM_CHECK_WITH_INFO(valid_length_tensor(accept_len),
                            "DSpARK confidence statistics require int32/int64 [B] accept lengths on the rates device");
    RTP_LLM_CHECK_WITH_INFO(!valid_lengths.defined() || valid_length_tensor(valid_lengths),
                            "DSpARK confidence statistics require int32/int64 [B] valid lengths on the rates device");
    RTP_LLM_CHECK_WITH_INFO(!eligible_rows.defined()
                                || (eligible_rows.dim() == 1 && eligible_rows.size(0) == batch_size
                                    && eligible_rows.scalar_type() == torch::kBool
                                    && eligible_rows.device() == confidence_rates.device()),
                            "DSpARK confidence statistics require bool [B] eligibility on the rates device");

    auto positions = torch::arange(gamma, confidence_rates.options().dtype(torch::kInt64)).unsqueeze(0);
    auto accepted  = accept_len.to(torch::kInt64).sub(1).unsqueeze(1);
    // The first rejection is observed; later positions are censored. When all
    // valid drafts were accepted, the bonus position is not a draft attempt.
    auto attempted = positions.le(accepted);
    if (valid_lengths.defined()) {
        attempted.logical_and_(positions.lt(valid_lengths.unsqueeze(1)));
    }
    if (eligible_rows.defined()) {
        attempted.logical_and_(eligible_rows.unsqueeze(1));
    }
    auto valid_prediction =
        torch::isfinite(confidence_rates).logical_and(confidence_rates.ge(0.0)).logical_and(confidence_rates.le(1.0));
    auto observed = attempted.logical_and(valid_prediction);
    auto invalid  = attempted.logical_and(valid_prediction.logical_not());

    // Select before arithmetic: multiplying an excluded NaN by zero would
    // otherwise poison a whole position's asynchronously aggregated metrics.
    auto zeros       = torch::zeros_like(confidence_rates);
    auto predictions = torch::where(observed, confidence_rates, zeros);
    auto labels      = torch::where(observed, positions.lt(accepted).to(torch::kFloat32), zeros);
    auto errors      = predictions - labels;
    return torch::stack({observed.to(torch::kFloat32).sum(0),
                         labels.sum(0),
                         predictions.sum(0),
                         errors.abs().sum(0),
                         errors.square().sum(0),
                         invalid.to(torch::kFloat32).sum(0)},
                        1);
}

SamplerOutput SpeculativeSampler::sampleDSparkDraft(const torch::Tensor& base_logits,
                                                    const torch::Tensor& anchors,
                                                    const torch::Tensor& temperature,
                                                    const torch::Tensor& top_k,
                                                    const torch::Tensor& top_p,
                                                    const torch::Tensor& markov_w1,
                                                    const torch::Tensor& markov_w2,
                                                    size_t               draft_vocab_size,
                                                    bool                 batch_all_greedy,
                                                    int                  force_topk,
                                                    int                  joint_max_top_k,
                                                    bool                 copy_token_ids,
                                                    const torch::Tensor& confidence_hidden_logits,
                                                    const torch::Tensor& confidence_head_w,
                                                    double               confidence_temperature) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.sample_dspark_draft");
    const bool compute_confidence = confidence_head_w.defined() || confidence_hidden_logits.defined();
    const auto batch_size         = compute_confidence ? temperature.numel() : 0;
    // Python projects the pre-norm proposal hidden to FP32 scalars. Finish
    // the head using the same previous-token embedding as the Markov bias.
    // No full hidden tensor crosses this API.
    RTP_LLM_CHECK_WITH_INFO(confidence_head_w.defined() == confidence_hidden_logits.defined(),
                            "DSpARK confidence head requires both hidden logits and the head weight");
    RTP_LLM_CHECK_WITH_INFO(!compute_confidence
                                || (std::isfinite(confidence_temperature) && confidence_temperature > 0.0),
                            "DSpARK confidence temperature must be positive and finite, got %g",
                            confidence_temperature);
    torch::Tensor confidence_rates;
    torch::Tensor hidden_logits;
    torch::Tensor confidence_w_markov;
    if (compute_confidence) {
        RTP_LLM_CHECK_WITH_INFO(
            confidence_hidden_logits.is_cuda() && confidence_hidden_logits.is_contiguous()
                && confidence_hidden_logits.scalar_type() == torch::kFloat32 && confidence_hidden_logits.dim() == 2
                && confidence_hidden_logits.size(0) == base_logits.size(0) && confidence_hidden_logits.size(1) == 1
                && confidence_hidden_logits.device() == base_logits.device(),
            "DSpARK confidence hidden logits must be contiguous CUDA FP32 [B*gamma, 1] "
            "on the logits device");
        RTP_LLM_CHECK_WITH_INFO(markov_w1.defined() && markov_w1.is_cuda() && markov_w1.dim() == 2
                                    && markov_w1.size(1) > 0 && markov_w1.device() == base_logits.device(),
                                "DSpARK confidence head requires CUDA [vocab, markov_rank] embeddings "
                                "on the logits device");
        const auto markov_rank = markov_w1.size(1);
        // The executor validates the exact hidden+rank width. This scalar API
        // can validate the full-head shape and consume its final rank columns.
        RTP_LLM_CHECK_WITH_INFO(confidence_head_w.is_cuda() && confidence_head_w.is_contiguous()
                                    && confidence_head_w.device() == base_logits.device()
                                    && confidence_head_w.dim() == 2 && confidence_head_w.size(0) == 1
                                    && confidence_head_w.size(1) > markov_rank
                                    && (confidence_head_w.scalar_type() == torch::kFloat32
                                        || confidence_head_w.scalar_type() == torch::kBFloat16),
                                "DSpARK confidence head weight must be contiguous CUDA FP32/BF16 "
                                "[1, hidden+markov_rank] on the logits device");
        confidence_w_markov = confidence_head_w.narrow(1, confidence_head_w.size(1) - markov_rank, markov_rank)
                                  .to(torch::kFloat32)
                                  .view({markov_rank, 1});
        hidden_logits    = confidence_hidden_logits.view({batch_size, static_cast<int64_t>(propose_step_), 1});
        confidence_rates = torch::empty({batch_size, static_cast<int64_t>(propose_step_)},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(base_logits.device()));
    }

    // Sampling graphs retain the fused Markov walk. The confidence for position
    // t depends on its pre-norm projection and token t-1, so it can be evaluated
    // from the finished proposal without changing sampling or RNG consumption.
    auto finish = [&](SamplerOutput output) {
        if (compute_confidence) {
            for (int64_t step = 0; step < static_cast<int64_t>(propose_step_); ++step) {
                auto previous_tokens = step == 0 ? anchors.reshape({batch_size}) : output.token_ids.select(1, step - 1);
                auto markov_embedding = markov_w1.index_select(0, previous_tokens.to(torch::kLong));
                auto markov_logit     = markov_embedding.to(torch::kFloat32).mm(confidence_w_markov);
                auto logit            = (hidden_logits.select(1, step) + markov_logit).squeeze(1);
                auto scaled           = (logit / static_cast<float>(confidence_temperature)).clamp(-60.0, 60.0);
                confidence_rates.select(1, step).copy_(torch::sigmoid(scaled));
            }
            output.confidence_rates = confidence_rates;
        }
        return output;
    };
#if USING_CUDA
    const auto& graphs = dspark_graphs_;
    if (graphs && batch_all_greedy && draft_vocab_size == graphs->vocab && base_logits.defined()
        && base_logits.device() == graphs->logits.device() && base_logits.scalar_type() == torch::kFloat32
        && base_logits.dim() == 2 && base_logits.is_contiguous() && base_logits.data_ptr() == graphs->logits_ptr
        && base_logits.size(1) == graphs->logits.size(1) && temperature.is_cuda()
        && temperature.scalar_type() == torch::kFloat32 && temperature.dim() == 1 && temperature.is_contiguous()
        && anchors.is_cuda() && anchors.scalar_type() == torch::kInt32 && anchors.is_contiguous()
        && anchors.numel() == temperature.numel() && anchors.device() == base_logits.device()
        && temperature.device() == base_logits.device()
        && base_logits.size(0) == temperature.numel() * static_cast<int64_t>(propose_step_)
        && sameGraphTensor(markov_w1, graphs->w1, graphs->w1_ptr)
        && sameGraphTensor(markov_w2, graphs->w2, graphs->w2_ptr)
        && sameGraphTensor(d2t_map_, graphs->map, graphs->map_ptr)) {
        const auto found = graphs->entries.find(temperature.numel());
        if (found != graphs->entries.end()) {
            const c10::cuda::CUDAGuard guard(base_logits.device());
            const auto                 stream = at::cuda::getCurrentCUDAStream();
            if (stream != graphs->replay_stream)
                graphs->replay_stream.synchronize();
            graphs->replay_stream    = stream;
            auto&              entry = *found->second;
            FusedD2DCopyParams copies;
            copies.add(anchors.data_ptr(), entry.anchors.data_ptr(), anchors.nbytes());
            copies.add(temperature.data_ptr(), entry.temperatures.data_ptr(), temperature.nbytes());
            fusedCopy(copies);
            entry.graph.replay();
            auto output = entry.output;
            if (copy_token_ids) {
                output.token_ids = output.token_ids.clone();
            }
            return finish(std::move(output));
        }
    }
#endif
    return finish(sampleDSparkDraftEager(base_logits,
                                         anchors,
                                         temperature,
                                         top_k,
                                         top_p,
                                         markov_w1,
                                         markov_w2,
                                         draft_vocab_size,
                                         batch_all_greedy,
                                         force_topk,
                                         joint_max_top_k));
}

SamplerOutput SpeculativeSampler::sampleDSparkDraftEager(const torch::Tensor& base_logits,
                                                         const torch::Tensor& anchors,
                                                         const torch::Tensor& temperature,
                                                         const torch::Tensor& top_k,
                                                         const torch::Tensor& top_p,
                                                         const torch::Tensor& markov_w1,
                                                         const torch::Tensor& markov_w2,
                                                         size_t               draft_vocab_size,
                                                         bool                 batch_all_greedy,
                                                         int                  force_topk,
                                                         int                  joint_max_top_k) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.sample_dspark_draft");
    RTP_LLM_CHECK_WITH_INFO(temperature.defined() && temperature.is_cuda() && temperature.is_contiguous()
                                && temperature.scalar_type() == torch::kFloat32 && temperature.dim() == 1,
                            "DSpARK draft temperatures must be contiguous CUDA FP32 [B]");
    const auto batch_size = temperature.numel();
    RTP_LLM_CHECK_WITH_INFO(base_logits.defined() && base_logits.is_cuda() && base_logits.is_contiguous()
                                && base_logits.scalar_type() == torch::kFloat32 && base_logits.dim() == 2
                                && base_logits.size(0) == batch_size * static_cast<int64_t>(propose_step_)
                                && base_logits.size(1) >= static_cast<int64_t>(draft_vocab_size),
                            "DSpARK C++ lm_head must emit contiguous CUDA FP32 [B*gamma,vocab_padded] logits with "
                            "vocab_padded >= draft vocab size");
    RTP_LLM_CHECK_WITH_INFO(anchors.defined() && anchors.is_cuda() && anchors.numel() == batch_size,
                            "DSpARK anchors must be a CUDA tensor with one token per request");
    RTP_LLM_CHECK_WITH_INFO(
        (!top_k.defined() || (top_k.is_cuda() && top_k.scalar_type() == torch::kInt32 && top_k.numel() == batch_size))
            && (!top_p.defined()
                || (top_p.is_cuda() && top_p.scalar_type() == torch::kFloat32 && top_p.numel() == batch_size)),
        "DSpARK draft top-k/top-p must be CUDA vectors with one value per request");

    // All-greedy verification accepts only exact target-token matches, so it
    // permits deterministic first-argmax proposals without dense q. Tiny-T
    // softmax can still distribute mass across ties or close logits; this
    // changes draft choices and RNG consumption, but target verification
    // preserves greedy output. Mixed batches retain the dense distribution.
    // Float partial indices are exact only below 2^24.
    const bool point_mass_path = batch_all_greedy && draft_vocab_size < (1u << 24);
#if USING_CUDA
    const int64_t compact_width =
        !point_mass_path && !top_k.defined() && force_topk > 0 && force_topk < static_cast<int64_t>(draft_vocab_size) ?
            force_topk :
            0;
    const bool sample_from_strided_q = !point_mass_path && compact_width == 0 && !top_k.defined() && !top_p.defined();
#else
    const int64_t compact_width         = 0;
    const bool    sample_from_strided_q = false;
#endif

    bool fused_greedy_output = false;
#if USING_CUDA
    fused_greedy_output = point_mass_path && propose_step_ > 0 && markov_w1.is_cuda()
                          && markov_w1.device() == base_logits.device() && markov_w1.dim() == 2
                          && markov_w1.scalar_type() == markov_w2.scalar_type()
                          && (!d2t_map_.defined()
                              || (d2t_map_.device() == base_logits.device() && d2t_map_.scalar_type() == torch::kInt64
                                  && d2t_map_.dim() == 1 && d2t_map_.numel() >= draft_vocab_size));
#endif
    auto token_matrix    = fused_greedy_output ? torch::empty({batch_size, static_cast<int64_t>(propose_step_)},
                                                           base_logits.options().dtype(torch::kInt32)) :
                                                 torch::Tensor();
    auto previous_tokens = anchors.reshape({batch_size});
    if (previous_tokens.scalar_type() != torch::kInt32 && previous_tokens.scalar_type() != torch::kInt64) {
        previous_tokens = previous_tokens.to(torch::kLong);
    }
    auto all_probabilities =
        point_mass_path ?
            torch::Tensor() :
            torch::empty({batch_size, static_cast<int64_t>(propose_step_), static_cast<int64_t>(draft_vocab_size)},
                         torch::TensorOptions().dtype(torch::kFloat32).device(base_logits.device()));
    if (compact_width > 0) {
        all_probabilities.zero_();
    }
    // Filtered paths need contiguous per-step scratch. Reuse it across steps
    // because each step fully rewrites it before sampling on the same stream.
    const auto sampling_width = compact_width > 0 ? compact_width : static_cast<int64_t>(draft_vocab_size);
    auto       sampling_probabilities =
        point_mass_path || sample_from_strided_q || compact_width > 0 ?
                  torch::Tensor() :
                  torch::empty({batch_size, sampling_width},
                         torch::TensorOptions().dtype(torch::kFloat32).device(base_logits.device()));
    const bool    filter_draft = !point_mass_path && (top_k.defined() || top_p.defined());
    torch::Tensor filtered_probabilities =
        filter_draft ? torch::empty({batch_size, sampling_width},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(base_logits.device())) :
                       torch::Tensor();
    torch::Tensor topk_probabilities = top_k.defined() && top_p.defined() && !point_mass_path && joint_max_top_k == 0 ?
                                           torch::empty_like(sampling_probabilities) :
                                           torch::Tensor();
    std::vector<torch::Tensor> token_columns;
    token_columns.reserve(propose_step_);
    // lm_head shards are padded to a TP alignment before gather. Sampling
    // must ignore those synthetic tail columns just like the regular target
    // sampler ignores padded vocabulary rows.
    auto proposal_logits =
        base_logits.narrow(1, 0, draft_vocab_size)
            .view({batch_size, static_cast<int64_t>(propose_step_), static_cast<int64_t>(draft_vocab_size)});

    torch::Tensor markov_embedding;
    for (int64_t step = 0; step < static_cast<int64_t>(propose_step_); ++step) {
        if (!markov_embedding.defined()) {
            markov_embedding = markov_w1.index_select(0, previous_tokens);
        }
        auto markov_bias = torch::mm(markov_embedding, markov_w2.transpose(0, 1));

#if USING_CUDA
        if (fused_greedy_output) {
            auto next_embedding = step + 1 < static_cast<int64_t>(propose_step_) ?
                                      torch::empty({batch_size, markov_w1.size(1)}, markov_w1.options()) :
                                      torch::Tensor();
            cudaMarkovArgmaxStep({proposal_logits.select(1, step), markov_bias, temperature},
                                 token_matrix.select(1, step),
                                 d2t_map_,
                                 markov_w1,
                                 next_embedding);
            markov_embedding = std::move(next_embedding);
            continue;
        }
#endif

        torch::Tensor sampled_draft_tokens;
        if (point_mass_path) {
            // Emit first-argmax int32 tokens without dense q or a sampling launch.
            sampled_draft_tokens = execMarkovArgmaxStep({proposal_logits.select(1, step), markov_bias, temperature});
        } else {
            // The draft sampling distribution must also be the q supplied to
            // rejection sampling. Match target top-k/top-p truncation here.
            torch::Tensor compact_indices;
            torch::Tensor draft_q;
#if USING_CUDA
            if (compact_width > 0) {
                auto step_logits = proposal_logits.select(1, step);
                auto selected =
                    compact_width == 4096 && batch_size >= 16 && draft_vocab_size == 129280
                            && step_logits.stride(0) % 256 == 0 && markov_bias.scalar_type() == torch::kBFloat16 ?
                        torch_ext::deepselect_fp32_markov_topk_prefilter(step_logits, markov_bias, temperature) :
                        torch_ext::deepselect_fp32_topk(
                            torch_ext::markov_scaled_logits(step_logits, markov_bias, temperature), compact_width);
                compact_indices = std::get<1>(selected);
                draft_q         = torch::softmax(std::get<0>(selected), -1);
            } else
#endif
            {
                execMarkovSoftmaxStep({proposal_logits.select(1, step),
                                       markov_bias,
                                       temperature,
                                       all_probabilities.select(1, step),
                                       sampling_probabilities});
                draft_q = sample_from_strided_q ? all_probabilities.select(1, step) : sampling_probabilities;
            }
            if (filter_draft) {
#if USING_CUDA
                const auto stream = reinterpret_cast<uintptr_t>(at::cuda::getCurrentCUDAStream().stream());
                if (joint_max_top_k > 0) {
                    torch_ext::deepselect_fp32_joint_topk_topp(
                        draft_q, filtered_probabilities, top_k, top_p, joint_max_top_k);
                } else if (top_k.defined() && top_p.defined()) {
                    top_k_renorm_probs(draft_q, topk_probabilities, top_k, 0, stream);
                    top_p_renorm_probs(draft_q, filtered_probabilities, top_p, 1.0, stream);
                    filtered_probabilities.masked_fill_(topk_probabilities.eq(0), 0.0);
                    filtered_probabilities.div_(filtered_probabilities.sum(-1, true).clamp_min_(1e-10));
                } else if (top_k.defined()) {
                    top_k_renorm_probs(draft_q, filtered_probabilities, top_k, 0, stream);
                } else {
                    top_p_renorm_probs(draft_q, filtered_probabilities, top_p, 1.0, stream);
                }
#else
                auto sorted = draft_q.sort(-1, true);
                auto values = std::get<0>(sorted);
                auto keep   = torch::ones_like(values, torch::TensorOptions().dtype(torch::kBool));
                if (top_k.defined()) {
                    auto kth = values.gather(1, top_k.clamp(1, sampling_width).to(torch::kLong).sub(1).unsqueeze(1));
                    keep.logical_and_(values.ge(kth));
                }
                if (top_p.defined()) {
                    auto nucleus = values.cumsum(-1).sub(values).lt(top_p.unsqueeze(1));
                    auto cutoff  = values.masked_fill(~nucleus, std::numeric_limits<float>::infinity()).amin(-1, true);
                    keep.logical_and_(values.ge(cutoff));
                }
                filtered_probabilities.zero_().scatter_(1, std::get<1>(sorted), values.masked_fill(~keep, 0.0));
                filtered_probabilities.div_(filtered_probabilities.sum(-1, true).clamp_min_(1e-10));
#endif
                draft_q = filtered_probabilities;
            }
            sampled_draft_tokens = execSampleFromProbs(draft_q).to(torch::kInt32);
            if (compact_width > 0) {
                all_probabilities.select(1, step).scatter_(1, compact_indices, draft_q);
                sampled_draft_tokens = compact_indices.gather(1, sampled_draft_tokens.to(torch::kLong).unsqueeze(1))
                                           .squeeze(1)
                                           .to(torch::kInt32);
            } else if (filter_draft) {
                all_probabilities.select(1, step).copy_(draft_q);
            }
        }
        auto sampled_target_tokens = sampled_draft_tokens;
        if (d2t_map_.defined()) {
            sampled_target_tokens = d2t_map_.index_select(0, sampled_draft_tokens.to(torch::kLong)).to(torch::kInt32);
        }
        token_columns.push_back(sampled_target_tokens);
        previous_tokens  = sampled_target_tokens.to(torch::kLong);
        markov_embedding = torch::Tensor();
    }

    SamplerOutput output;
    output.token_ids = fused_greedy_output ? token_matrix : torch::stack(token_columns, 1).contiguous();
    if (!point_mass_path) {
        output.all_probs = std::move(all_probabilities);
    }
    output.token_ids_are_point_mass = point_mass_path;
    return output;
}

SpeculativeSamplerOutput SpeculativeSampler::forward(const std::list<GenerateStreamPtr>& streams,
                                                     SamplerOutput&                      draft_sampler_output,
                                                     SamplerOutput&                      target_sampler_output) {
    // TensorHolder release point (SpeculativeSampler): advances host tensors
    // staged for rejection sampling H2D in the previous forward.
    buffer_holder_.release();
    SpeculativeSamplerOutput sample_output;
    batchSample(sample_output, streams, draft_sampler_output, target_sampler_output);

    return sample_output;
}

void SpeculativeSampler::batchSample(SpeculativeSamplerOutput&           sample_output,
                                     const std::list<GenerateStreamPtr>& streams,
                                     SamplerOutput&                      draft_sampler_output,
                                     SamplerOutput&                      target_sampler_output) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample");
    torch::Device target_device = getTorchCudaDevice();

    int batch_size = streams.size();

    const auto& valid_draft_lengths = draft_sampler_output.valid_draft_lengths;
    if (valid_draft_lengths.defined()) {
        RTP_LLM_CHECK_WITH_INFO(
            valid_draft_lengths.is_cuda()
                && valid_draft_lengths.get_device() == cuda_graph::graphGetCurrentStream().device_index()
                && valid_draft_lengths.scalar_type() == torch::kInt32 && valid_draft_lengths.dim() == 1
                && valid_draft_lengths.size(0) == batch_size && valid_draft_lengths.is_contiguous(),
            "DSpARK valid draft lengths must be contiguous CUDA int32 [B] on the sampling device");
        // Legacy finalization assumes full width, including forced acceptance
        // and row-gamma bonus selection. Only the CUDA folded path supports L.
        RTP_LLM_CHECK_WITH_INFO(execRejectionSamplingSupportsFinalize(),
                                "DSpARK bounded verification requires CUDA folded finalization support");
    }

    auto draft_token_ids = draft_sampler_output.token_ids;

    auto       draft_token_probs      = draft_sampler_output.all_probs;
    auto       target_token_probs     = target_sampler_output.all_probs;
    const bool draft_probs_point_mass = draft_sampler_output.token_ids_are_point_mass;
    // Greedy top_k==1 verification may skip materializing the target one-hot
    // matrix; the rejection kernel then synthesizes q from target_token_ids
    // (bit-identical to the matrix). target_token_probs stays defined with its
    // [B, steps+1, vocab] shape for the launch-config validation below.
    const bool target_probs_point_mass = target_sampler_output.token_ids_are_point_mass;

    buffer_holder_.hold_host(draft_token_ids);
    auto draft_token_ids_d_t = draft_token_ids.to(target_device, true);

    auto target_token_ids_d_t = target_sampler_output.token_ids;
    if (!target_token_ids_d_t.is_cuda()) {
        buffer_holder_.hold_host(target_token_ids_d_t);
        target_token_ids_d_t = target_token_ids_d_t.to(target_device, true);
    }

    // Both per-stream masks share one pinned allocation and one device copy.
    auto  masks = torch::empty({2, (long)batch_size}, torch::TensorOptions().dtype(torch::kBool).pinned_memory(true));
    auto* do_sample_ptr  = masks.data_ptr<bool>();
    auto* force_mask_ptr = do_sample_ptr + batch_size;
    bool  has_force      = false;
    int   stream_idx     = 0;
    for (const GenerateStreamPtr& stream : streams) {
        do_sample_ptr[stream_idx]  = stream->generateConfig()->stochastic();
        force_mask_ptr[stream_idx] = stream->forceSpAccept();
        has_force |= force_mask_ptr[stream_idx];
        stream_idx++;
    }
    buffer_holder_.hold_host(masks);

    // Folded finalization capability (CUDA kernel): the rejection kernel
    // applies the force-accept override and the -1 -> 0 padding reset itself
    // and writes every output element on every path, so the output buffers
    // need no zero init and the legacy eager post-rejection chain
    // (full_like + cat + where x2 + eq + masked_fill) is skipped entirely.
    // ROCm/generic return false and keep the legacy chain byte-for-byte.
    const bool finalize_in_kernel = execRejectionSamplingSupportsFinalize();

    // An undefined force mask retains the ordinary rejection path.
    auto          rand_options      = torch::TensorOptions().device(target_device).dtype(torch::kFloat);
    torch::Tensor uniform_samples_d = torch::rand({(long)batch_size, (long)propose_step_ + 1}, rand_options);

    // Override per-stream uniform samples with seeded generator when random_seed is set,
    // ensuring deterministic acceptance for reproducible iter_count.
    {
        int idx = 0;
        for (const auto& stream : streams) {
            auto gen = stream->getGenerator();
            if (gen.defined()) {
                uniform_samples_d[idx] = torch::rand({(long)propose_step_ + 1}, gen, std::nullopt, rand_options);
            }
            idx++;
        }
    }

    if (replayRejectionGraph(sample_output,
                             masks,
                             draft_token_ids_d_t,
                             target_token_ids_d_t,
                             uniform_samples_d,
                             target_token_probs,
                             draft_probs_point_mass && target_probs_point_mass,
                             valid_draft_lengths))
        return;

    auto       masks_d      = masks.to(target_device, true);
    auto       do_sample_d  = masks_d.select(0, 0);
    const auto force_mask   = has_force ? masks_d.select(0, 1) : torch::Tensor();
    auto       force_mask_d = finalize_in_kernel ? force_mask : torch::Tensor();

    auto draft_token_probs_d_t  = draft_token_probs;
    auto target_token_probs_d_t = target_token_probs;
    // With the folded finalize the kernel writes every element of both output
    // buffers on every path (forced rows included), so the zero inits are dead
    // stores; without it the legacy kernels/ops keep their zero-init contract.
    auto int_out_options = torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false);
    torch::Tensor output_token_ids_d          = finalize_in_kernel ?
                                                    torch::empty({(long)batch_size, (long)propose_step_ + 1}, int_out_options) :
                                                    torch::zeros({(long)batch_size, (long)propose_step_ + 1}, int_out_options);
    torch::Tensor output_accepted_token_num_d = finalize_in_kernel ? torch::empty({(long)batch_size}, int_out_options) :
                                                                     torch::zeros({(long)batch_size}, int_out_options);

    RTP_LLM_CHECK_WITH_INFO(draft_probs_point_mass
                                || (draft_token_probs_d_t.defined() && draft_token_probs_d_t.dim() == 3),
                            "draft probabilities must be [B, steps, vocab] unless token ids define a point mass");
    if (!draft_probs_point_mass && draft_token_probs_d_t.size(2) != target_token_probs_d_t.size(2)) {
        const int64_t target_vocab_size = target_token_probs_d_t.size(2);
        const int64_t num_spec          = draft_token_probs_d_t.size(1);

        // Reuse pre-allocated padding buffer to avoid per-forward GPU allocation.
        // Grow-only along batch / num_spec dims; vocab dim must match exactly.
        const bool need_realloc = !draft_probs_padding_buffer_.defined()
                                  || draft_probs_padding_buffer_.size(0) < (int64_t)batch_size
                                  || draft_probs_padding_buffer_.size(1) < num_spec
                                  || draft_probs_padding_buffer_.size(2) != target_vocab_size
                                  || draft_probs_padding_buffer_.dtype() != draft_token_probs_d_t.dtype()
                                  || draft_probs_padding_buffer_.device() != draft_token_probs_d_t.device();
        if (need_realloc) {
            const int64_t cap_b =
                std::max((int64_t)batch_size,
                         draft_probs_padding_buffer_.defined() ? draft_probs_padding_buffer_.size(0) : (int64_t)0);
            const int64_t cap_s = std::max(
                num_spec, draft_probs_padding_buffer_.defined() ? draft_probs_padding_buffer_.size(1) : (int64_t)0);
            draft_probs_padding_buffer_ =
                torch::zeros({cap_b, cap_s, target_vocab_size}, draft_token_probs_d_t.options());
        }

        auto draft_probs_padding = draft_probs_padding_buffer_.narrow(0, 0, (int64_t)batch_size).narrow(1, 0, num_spec);
        draft_probs_padding.zero_();
        draft_probs_padding.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), d2t_map_},
                                       draft_token_probs_d_t);
        draft_token_probs_d_t = draft_probs_padding;
    }

    {
        RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.execRejectionSampling");
        execRejectionSampling({
            draft_token_probs_d_t,
            draft_token_ids_d_t,
            uniform_samples_d,
            target_token_probs_d_t,
            target_token_ids_d_t,
            output_token_ids_d,
            output_accepted_token_num_d,
            do_sample_d,
            draft_probs_point_mass,
            target_probs_point_mass,
            force_mask_d,
            finalize_in_kernel,
            valid_draft_lengths,
        });
    }

    RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.post_rejection_sampling");

    if (!finalize_in_kernel) {
        // Legacy post-rejection finalization (ROCm/generic): force-accept
        // override + -1 -> 0 padding reset, byte-for-byte the historical
        // eager chain. The CUDA kernel folds this when finalize_in_kernel.

        // forceSpAccept: override rejection sampling results for streams that
        // requested forced acceptance — accept all draft tokens plus the
        // target bonus token.
        {
            if (force_mask.defined()) {
                RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.post_rejection_sampling.forceSpAccept");
                // target_token_ids_d_t layout: [batch_size * (propose_step+1), token_stride]
                // Extract the bonus token at position propose_step for each batch item.
                int64_t token_stride = target_token_ids_d_t.size(1);
                auto    target_bonus_t =
                    target_token_ids_d_t.reshape({(long)batch_size, (long)(propose_step_ + 1), token_stride});
                auto target_bonus = target_bonus_t.select(1, propose_step_).select(1, token_stride - 1).unsqueeze(1);
                // forced_tokens: draft tokens [0..propose_step-1] + target bonus
                auto forced_tokens = torch::cat({draft_token_ids_d_t, target_bonus}, 1);
                auto force_mask_2d = force_mask.unsqueeze(1).expand_as(output_token_ids_d);
                output_token_ids_d = torch::where(force_mask_2d, forced_tokens, output_token_ids_d);
                output_accepted_token_num_d =
                    torch::where(force_mask,
                                 torch::full_like(output_accepted_token_num_d, (int32_t)(propose_step_ + 1)),
                                 output_accepted_token_num_d);
            }
        }

        // use async sample here, we assume accept all tokens
        // so we need to reset -1 to 0 in output_token_ids_d
        output_token_ids_d.index_put_({output_token_ids_d == -1}, 0);
    }

    sample_output.accept_tokens = output_token_ids_d;
    sample_output.accept_len    = output_accepted_token_num_d;

    sample_output.accept_tokens_cpu = sample_output.accept_tokens.to(torch::kCPU, true);
    sample_output.accept_len_cpu    = sample_output.accept_len.to(torch::kCPU, true);
    sample_output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams,
                                      SamplerOutput&                      draft_sampler_output,
                                      SamplerOutput&                      target_sampler_output) const {}

}  // namespace speculative
}  // namespace rtp_llm
