#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "ATen/ops/cross_entropy_loss.h"
#include "c10/util/Optional.h"
#include "rtp_llm/cpp/core/TrackerAllocator.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/RpcErrorCode.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "torch/extension.h"
#include "torch/types.h"
#include <numeric>

using namespace std;
using namespace rtp_llm;

namespace rtp_llm {

DeviceBase::DeviceBase(const DeviceInitParams& params)
    : device_id_(params.device_id)
    , init_params_(params)
    {
        // 默认stdout输出到文件的逻辑是全缓冲，导致ft_log和autil_log日志刷不出来，手动设置为行缓冲
        setlinebuf(stdout);
    }

void DeviceBase::init() {
    buffer_manager_.reset(new BufferManager(getAllocator(), getHostAllocator()));
    static char* enable_device_perf_env_char = std::getenv("ENABLE_DEVICE_PERF");
    enable_device_perf_ = (enable_device_perf_env_char != nullptr && std::string(enable_device_perf_env_char) == "ON") ? true : false;
}

void DeviceBase::setTraceMemory(bool trace_memory) {
    buffer_manager_->setTraceMemory(trace_memory);
}

void DeviceBase::holdBufferRecycle() {
    buffer_manager_->holdRecycle();
}

void DeviceBase::releaseBufferRecycleHold() {
    buffer_manager_->releaseRecycleHold();
}

std::shared_ptr<rtp_llm::CacheStore> DeviceBase::cacheStore() {
    return cache_store_;
}

MemoryStatus DeviceBase::getDeviceMemoryStatus() {
    return MemoryStatus();
}

DeviceStatus DeviceBase::getDeviceStatus() {
    DeviceStatus status;

    status.device_memory_status = getDeviceMemoryStatus();

    const auto buffer_status = queryBufferStatus();
    status.device_memory_status.allocated_bytes = buffer_status.device_allocated_bytes;
    status.device_memory_status.preserved_bytes = buffer_status.device_preserved_bytes;
    status.device_memory_status.available_bytes = status.device_memory_status.free_bytes + status.device_memory_status.preserved_bytes;
    status.device_memory_status.max_consumed_bytes = buffer_status.device_max_consumed_bytes;
    status.host_memory_status.allocated_bytes = buffer_status.host_allocated_bytes;
    return status;
}

void DeviceBase::traceMemoryUsage() {
    RTP_LLM_LOG_INFO("Device Memory: %s", buffer_manager_->printAllocationRecords(getAllocator()).c_str());
    RTP_LLM_LOG_INFO("Host Memory: %s", buffer_manager_->printAllocationRecords(getHostAllocator()).c_str());
    return;
}

AllocationType DeviceBase::getMemAllocationType(const MemoryType type) {
    return (type == getAllocator()->memoryType()) ? AllocationType::DEVICE : AllocationType::HOST;
}

BufferStatus DeviceBase::queryBufferStatus() {
    return buffer_manager_->queryStatus();
}

BufferPtr DeviceBase::allocateBuffer(const BufferParams& params, const BufferHints& hints) {
    return buffer_manager_->allocate(params, hints);
}

BufferPtr DeviceBase::allocateBufferLike(const Buffer& buffer,
                                         const AllocationType atype,
                                         const BufferHints& hints) {
    if (buffer.isQBuffer()) {
        auto kernel = allocateBufferLike((reinterpret_cast<const QBuffer*>(&buffer)->kernel()),
                                         atype,
                                         hints);
        auto scales = allocateBufferLike((reinterpret_cast<const QBuffer*>(&buffer)->scales()),
                                         atype,
                                         hints);
        auto zeros = allocateBufferLike((reinterpret_cast<const QBuffer*>(&buffer)->zeros()),
                                        atype,
                                        hints);
        return BufferPtr(new QBuffer(std::move(kernel),
                                     std::move(scales),
                                     std::move(zeros)));
    }
    return allocateBuffer({buffer.type(), buffer.shape(), atype}, hints);
}

void DeviceBase::checkError() {
    return;
}

void DeviceBase::syncAndCheck() {
    return;
}

void DeviceBase::syncDeviceStream(DeviceStream stream) {
    return;
}

DevicePrepOutput DeviceBase::prepareModelRun(const DevicePrepParams& params) {
    return DevicePrepOutput();
}

void DeviceBase::syncCommunication(bool timeout) {
    return;
}

void DeviceBase::syncCommunication(ParallelMode mode, bool timeout) {
    return;
}

void DeviceBase::overlappedCommBarrier() {
    syncCommunication();
}

DeviceHookPtr DeviceBase::createCommHook() {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceBase::overlappedComputeBarrier() {
    syncCommunication();
}

DeviceEventPtr DeviceBase::createEvent() {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

DeviceEventPtr DeviceBase::createTorchEvent() {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceBase::setCacheStore(std::shared_ptr<rtp_llm::CacheStore> cache_store) {
    cache_store_ = cache_store;
}

void DeviceBase::writeCacheStore(const WriteCacheParams& params) {
    auto& param = params.common;
    if (param.warmup) {
        RTP_LLM_LOG_DEBUG("is warmup, so ignore writeCacheStore");
        return;
    }
    if (!param.pd_separation || param.context_batch_size == 0) {
        RTP_LLM_LOG_DEBUG("pd_separation = %d, context_batch_size = %d, so ignore writeCacheStore",
            param.pd_separation, param.context_batch_size);
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(param.cache_store_inputs.has_value()
                        && param.cache_store_inputs->host_kv_cache_offset, "failed to get host_kv_cache_offset");
    const auto max_blocks_per_batch = param.cache_store_inputs->host_kv_cache_offset->shape()[1];
    const auto seq_size_per_block  = params.configs.tokens_per_block;
    auto offset_addr = param.cache_store_inputs->host_kv_cache_offset->data<int32_t>();
    auto k_cache_data = (uint64_t*)param.kv_cache->k_cache_buffer->data();
    auto v_cache_data = (uint64_t*)param.kv_cache->v_cache_buffer->data();
    // auto k_scale_data = (uint64_t*)(param.kv_cache->k_scale_buffer ? param.kv_cache->k_scale_buffer->data() : nullptr);
    // auto v_scale_data = (uint64_t*)(param.kv_cache->v_scale_buffer ? param.kv_cache->v_scale_buffer->data() : nullptr);

    RTP_LLM_CHECK_WITH_INFO(param.context_batch_size == param.request_pd_separation->size(), "size not same");
    RTP_LLM_CHECK_WITH_INFO(param.context_batch_size == param.request_id->size(),
                        "context batch size and request id size is not same");

    RTP_LLM_LOG_DEBUG("write cache store, context_batch_size is %ld", param.context_batch_size);

    for (size_t batch_id = 0; batch_id < param.context_batch_size; batch_id++) {
        if (*(param.request_pd_separation->dataWithOffset<bool>(batch_id)) == false) {
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(param.cache_store_inputs.has_value()
                            && param.cache_store_inputs->prefix_lengths_host
                            && param.cache_store_inputs->input_lengths_host,
                            "failed to get prefix_length_host and input_length_host for cache store");
        RTP_LLM_CHECK_WITH_INFO(param.cache_store_inputs->prefix_lengths_host->data<int>()[batch_id] % seq_size_per_block == 0,
                            "prefix_length \% seq_size_per_block != 0");
        int reuse_block_num = param.cache_store_inputs->prefix_lengths_host->data<int>()[batch_id] / seq_size_per_block;
        int block_num = (param.cache_store_inputs->input_lengths_host->data<int>()[param.decoder_batch_size + batch_id]
                            + seq_size_per_block - 1) / seq_size_per_block;
        auto request_id = *(param.request_id->dataWithOffset<int64_t>(batch_id));
        auto request_blocks = std::make_shared<RequestBlockBuffer>(std::to_string(request_id), createEvent());
        RTP_LLM_LOG_DEBUG("write cache store, blocks num is %ld", block_num + reuse_block_num);
        for (size_t index = 0; index < block_num + reuse_block_num; index++) {
            auto cache_key = makeCacheKey(params.common.model_id, param.cache_keys[batch_id * max_blocks_per_batch + index], param.layer_id);
            // FT_LOG_DEBUG("write kv cache_key %s", cache_key.c_str());
            auto block_id = *(offset_addr + (param.decoder_batch_size + batch_id) * max_blocks_per_batch + index);
            void* k_addr = (void*)((int8_t*)k_cache_data + block_id * param.k_block_size);
            std::shared_ptr<void> k_block_addr(k_addr, [](void* p) { });
            request_blocks->addBlock("k_" + cache_key, k_block_addr, param.k_block_size, true, true);
            // if (k_scale_data) {
            //     void* k_scale_addr = (void*)((int8_t*)k_scale_data + block_id * param.scale_block_size);
            //     std::shared_ptr<void> k_scale_block_addr(k_scale_addr, [](void* p) { });
            //     request_blocks->addBlock("k_scale" + cache_key, k_scale_block_addr, param.scale_block_size, true, true);
            // }
            // mla kvcache 不存储 v_cache
            if (params.mla_kvcache) {
                continue;
            }
            void* v_addr = (void*)((int8_t*)v_cache_data + block_id * param.v_block_size);
            std::shared_ptr<void> v_block_addr(v_addr, [](void* p) { });
            request_blocks->addBlock("v_" + cache_key, v_block_addr, param.v_block_size, true, true);
            // if (v_scale_data) {
            //     void* v_scale_addr = (void*)((int8_t*)v_scale_data + block_id * param.scale_block_size);
            //     std::shared_ptr<void> v_scale_block_addr(v_scale_addr, [](void* p) { });
            //     request_blocks->addBlock("v_scale" + cache_key, v_scale_block_addr, param.scale_block_size, true, true);
            // }
        }
        auto storeCallback = [layer_id = param.layer_id, request_id](bool success, CacheStoreErrorCode ec) {
            if (!success) {
                RTP_LLM_LOG_WARNING("query [%ld], layer id [%d], "
                               "call store kv cache failed, ec is %d, error msg is [%s]",
                               request_id, layer_id, ec, ErrorCodeToString(transCacheStoreErrorCode(ec)).c_str());
            }
        };
        cache_store_->store(request_blocks, storeCallback);
    }
}

CloneOutput DeviceBase::clone(const CloneParams& params) {
    const auto& src = params.input;
    auto dst = allocateBufferLike(src, params.alloc_type, params.hints);
    copy({*dst, src, params.overlapped});
    return dst;
}

SelectOutput DeviceBase::select(const SelectParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.dim < params.input.shape().size(),
                          "Select dim %ld out of range with input shape %s.",
                          params.dim, params.input.debugString().c_str());
    RUNTIME_ASSERT_OP_ARG(params.index.shape().size() == 1, "Select index must be 1D.");
    RUNTIME_ASSERT_OP_ARG(params.index.type() == DataType::TYPE_INT32, "Select index must be int32.");
    RUNTIME_ASSERT_OP_ARG(params.index.where() != MemoryType::MEMORY_GPU, "Select index must on CPU.");

    const auto& src = params.input;
    const auto& idx_buf = params.index;
    const auto dim = params.dim;
    auto selected_shape = src.shape();
    selected_shape[dim] = idx_buf.shape()[0];
    auto selected = allocateBuffer({src.type(), selected_shape, getMemAllocationType(src.where())});

    const int pre_select_size = std::accumulate(
        selected_shape.begin(), selected_shape.begin() + dim, 1UL, std::multiplies<size_t>());
    const auto post_select_stride = (int32_t)std::accumulate(
        selected_shape.begin() + dim + 1, selected_shape.end(), 1UL, std::multiplies<size_t>());

    // both src and dst needs to be viewed into 1-d buffer.
    auto src_view = src.reshape({src.size()});
    auto dst_view = selected->reshape({selected->size()});

    for (auto i = 0; i < int(idx_buf.shape()[0]); i++) {
        const auto idx = idx_buf.data<int32_t>()[i];
        for (auto j = 0; j < pre_select_size; j++) {
            const auto src_offset = j * src.shape()[dim] * post_select_stride + idx * post_select_stride;
            const auto dst_offset = j * idx_buf.size() * post_select_stride + i * post_select_stride;
            copy({dst_view.view(dst_offset, post_select_stride), src_view.view(src_offset, post_select_stride)});
        }
    }

    return selected;
}

ConcatOutput DeviceBase::concat(const ConcatParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.dim == 0, "Concat only support dim 0, but got %lu.", params.dim);
    RUNTIME_ASSERT_OP_ARG(params.inputs.size() > 0, "Concat requires at least 1 input.");
    if (params.inputs.size() == 1) {
        return params.inputs[0];
    }

    const auto concated_length = std::accumulate(
        params.inputs.begin(), params.inputs.end(), 0UL,
        [](size_t sum, const BufferPtr& buffer) {
            return sum + buffer->shape()[0];
        });
    auto concated_shape = params.inputs[0]->shape();
    concated_shape[0] = concated_length;
    const auto type = params.inputs[0]->type();
    auto concated = allocateBuffer({
        type, concated_shape, getMemAllocationType(params.inputs[0]->where())});

    size_t offset = 0;
    for (int i = 0; i < int(params.inputs.size()); i++) {
        const auto& input = params.inputs[i];
        const auto& shape = input->shape();
        RUNTIME_ASSERT_OP_ARG(
            shape.size() == concated_shape.size(),
            "Concat input [%d] shape size %ld does not match concated shape size %lu.",
            i, shape.size(), concated_shape.size());
        for (int j = 1; j < int(concated_shape.size()); j++) {
            RUNTIME_ASSERT_OP_ARG(
                shape[j] == concated_shape[j],
                "Concat input [%d] shape[%d] %ld does not match concated shape[%d] %ld.",
                i, j, shape[j], j, concated_shape[j]);
        }
        RUNTIME_ASSERT_OP_ARG(
            input->type() == type,
            "Concat input [%d] type %d does not match concated type %d.",
            i, input->type(), type);

        copy({concated->view(offset, (int64_t)shape[0]), *input});
        offset += shape[0];
    }
    return concated;
}

SplitOutput DeviceBase::split(const SplitParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.dim < params.input.dim()
                              && std::accumulate(params.split_sizes.begin(), params.split_sizes.end(), 0)
                                     == params.input.shape()[params.dim],
                          "split params args error, dim [%ld] split_size_sum [%d] input[%s]",
                          params.dim,
                          std::accumulate(params.split_sizes.begin(), params.split_sizes.end(), 0),
                          params.input.debugString().c_str());
    RUNTIME_ASSERT_OP_ARG(!params.overlapped, "split base impl not support overlap");

    torch::Tensor              input_t = Buffer2torchTensor(params.input, false);
    at::IntArrayRef            split_sizes((int64_t*)params.split_sizes.data(), params.split_sizes.size());
    std::vector<torch::Tensor> outputs_t = input_t.split_with_sizes(split_sizes, params.dim);
    assert(params.split_sizes.size() == outputs_t.size());
    std::vector<BufferPtr> outputs;
    for (int i = 0; i < params.split_sizes.size(); ++i) {
        outputs.emplace_back(clone({*torchTensor2Buffer(outputs_t[i].contiguous())}));
    }
    return {outputs};
}

LossOutput DeviceBase::loss(const LossParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.logits.where() == params.labels.where(), "logits and labels must be same device, but got %d and %d.", (int)params.logits.where(), (int)params.labels.where());
    RUNTIME_ASSERT_OP_ARG(params.logits.shape()[0] == params.labels.shape()[0], "logits and labels must be same dim0, but got %d and %d.", (int)params.logits.shape()[0], (int)params.labels.shape()[0]);
    torch::Tensor logits = Buffer2torchTensor(params.logits, false);
    torch::Tensor labels = Buffer2torchTensor(params.labels, false).toType(torch::kInt64);
    torch::Tensor output;
    output = torch::cross_entropy_loss(logits, labels, torch::nullopt, at::Reduction::None).to(torch::TensorOptions(torch::kFloat32));
    return clone({*torchTensor2Buffer(output)});
}

MaskOutput DeviceBase::attentionMask(const MaskParams& params) {
    const int *input_lengths = params.input_lengths.data<int32_t>();
    const int batch_size = params.input_lengths.size();
    const int max_input_seq_len = *std::max_element(input_lengths, input_lengths + batch_size);
    const auto torch_type = dataTypeToTorchType(params.dtype);
    auto tensor_options = torch::TensorOptions(torch::kBool).device(torch::Device(torch::kCPU));
    auto attention_mask = torch::ones({(int)max_input_seq_len, (int)max_input_seq_len}, tensor_options);
    if (params.is_causal) {
        attention_mask = attention_mask.tril();
    }
    attention_mask = attention_mask.unsqueeze_(0).tile({(int)batch_size, 1, 1}).to(torch_type);
    for (int i = 0; i < batch_size; ++i) {
        attention_mask[i].slice(0, input_lengths[i], max_input_seq_len) = 0;
        if (!params.is_causal) {
            attention_mask[i].slice(1, input_lengths[i], max_input_seq_len) = 0;
        }
    }
    if (params.prefix_lengths.size()) {
        RTP_LLM_CHECK(int(params.prefix_lengths.size()) == batch_size);
        const int *prefix_lengths = params.prefix_lengths.data<int32_t>();
        auto max_reuse_length = *std::max_element(prefix_lengths, prefix_lengths + batch_size);
        attention_mask = torch::cat({attention_mask, torch::zeros({(int)batch_size, max_input_seq_len, max_reuse_length}).to(torch_type)}, -1);
        if (max_reuse_length) {
            for (int i = 0; i < batch_size; ++i) {
                attention_mask[i] = attention_mask[i].roll({prefix_lengths[i]}, {-1});
                attention_mask[i].slice(0, 0, input_lengths[i]).slice(1, 0, prefix_lengths[i]) = 1;
            }
        }
    }
    return clone({*torchTensor2Buffer(attention_mask)});
}

MultimodalEmbeddingOutput DeviceBase::multimodalEmbedding(const MultimodalEmbeddingParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.multimodal_locs, "no multimodal input location found");
    const auto& embeddings = params.word_embeddings;
    const auto& features = params.multimodal_features.value().get();
    const auto& multimodal_locs = params.multimodal_locs.value().get();
    const auto mm_num = features.size();

    RUNTIME_ASSERT_OP_ARG(
        embeddings->typeSize() == features[0]->typeSize(),
        "type size of embeddings and multimodal features should be equal.");

    for (int i = 0; i < mm_num; ++i) {
        auto& feature = features[i];
        auto loc = multimodal_locs.dataWithOffset<int32_t>(i);
        copy({embeddings->view(*loc, feature->shape()[0]), *feature});
    }

    return move(embeddings);
}


AllReduceOutput DeviceBase::allReduce(const AllReduceParams& params) {
    if (getDeviceProperties().tp_size == 1) {
        return AllReduceOutput({params.buffer});
    };
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void DeviceBase::prepareCommBuffer(const PrepareCommBufferParams& params) {}

OverallExpertStats DeviceBase::createMoeExpertStates(const ExpertStatsParams& params)
{
    OverallExpertStats states;
    auto const layer_num = params.layer_num;
    auto const logic_expert_num = params.log_exp_num;
    auto const physic_expert_num = params.phy_exp_num;
    auto const ep_size = params.ep_size;

    states.layer_num = layer_num;
    states.ep_size = ep_size;
    states.log_exp_num = logic_expert_num;
    states.phy_exp_num = physic_expert_num;

    auto logic_buff = allocateBuffer({DataType::TYPE_INT32, {layer_num, logic_expert_num}, AllocationType::DEVICE}, {"exp_log_cnt"});

    auto gpu_load_buff = allocateBuffer({DataType::TYPE_INT32, {layer_num, ep_size}, AllocationType::DEVICE}, {"phy_gpu_load"});

    states.stats_buf.log_stats_buf = logic_buff;
    states.stats_buf.gpu_loads_buf = gpu_load_buff;


    cleanMoeExpertStates(states);

    return states;
}

void DeviceBase::cleanMoeExpertStates(const OverallExpertStats& stats)
{
    bufMemset(*stats.stats_buf.log_stats_buf, 0);
    bufMemset(*stats.stats_buf.gpu_loads_buf, 0);
}

void DeviceBase::updateExpertGpuLoads(const MoeConfigs& moe_conf, const OptionalExpertStats& expert_stats, BufferPtr expert_ids) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

} // namespace rtp_llm
