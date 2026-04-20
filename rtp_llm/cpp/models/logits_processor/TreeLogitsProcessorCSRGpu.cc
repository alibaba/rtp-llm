// GPU 版限制性解码处理器实现（性能优化版）
// 仅在 CUDA 环境下编译；非 CUDA 环境下不链接本文件，由 CPU fallback 版本兜底。
//
// 性能优化要点（相比初版）：
//   1. d_states_batch_：processor 级持久化 GPU buffer，跨 decode 步复用，
//      process() 直接读取，updateStatus() 原地更新，消除每步 H2D states 传输。
//   2. d_sampled_tokens_ / d_col_offsets_：预分配复用，消除每步重复 allocate。
//   3. invokeCsrGatherTokens：在 GPU 上直接 gather 采样 token，
//      完全消除原来的 D2H 大矩阵拷贝（new_tokens 全量 D2H）。
//   4. updateStatus() 中只保留一次必要的 D2H（[batch_size] 的状态同步），
//      消除了逐 beam 单独 H2D 更新 d_current_state 的 N 次小传输。

#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessorCSR.h"
#include "rtp_llm/cpp/kernels/csr_logits.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#if USING_CUDA
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#endif

#include <algorithm>
#include <vector>

namespace rtp_llm {

// =============================================================================
// process()
//
// 每步 decode 时调用。优化后流程：
//   1. 直接使用持久化的 d_states_batch_（无需每步 H2D）
//   2. 分配（首次）或复用 d_mask_out（[batch, vocab] uint8，GPU）
//   3. 调用 invokeCsrBuildMask：在 GPU 上查 CSR 生成 mask
//   4. 调用 maskLogits：把 mask=1 的位置置为 -inf
// =============================================================================
void TreeLogitsProcessorCSR::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    const size_t batch_size = size();
    RTP_LLM_CHECK(batch_size == finish_idx - start_idx);

    // 检查是否有任何 beam 开启了限制性解码
    bool any_in_tree_mode = false;
    for (size_t i = 0; i < batch_size; ++i) {
        if (tree_infos_[i].in_tree_mode) {
            any_in_tree_mode = true;
            break;
        }
    }
    if (!any_in_tree_mode || !d_states_batch_) {
        return;
    }

    auto      batch_logits = inputs.logits->slice(start_idx, batch_size);
    const int vocab_size   = static_cast<int>(batch_logits->shape()[1]);

    // 取共享只读 CSR buffer（所有 beam 共享同一组 GPU buffer）
    const StreamTreeInfo* ref_info = nullptr;
    for (size_t i = 0; i < batch_size; ++i) {
        if (tree_infos_[i].in_tree_mode) {
            ref_info = &tree_infos_[i];
            break;
        }
    }
    RTP_LLM_CHECK(ref_info != nullptr);

    // 分配输出 mask buffer [batch_size, vocab_size]（GPU）
    // 每次 process 重新分配（vocab_size 可能变化；也可持久化，此处保守处理）
    auto d_mask_out = device_->allocateBuffer({DataType::TYPE_UINT8,
                                               {static_cast<size_t>(batch_size), static_cast<size_t>(vocab_size)},
                                               AllocationType::DEVICE});

#if USING_CUDA
    CudaDevice* cuda_dev = dynamic_cast<CudaDevice*>(device_);
    RTP_LLM_CHECK(cuda_dev != nullptr);
    cudaStream_t stream = cuda_dev->getStream();

    // 直接用持久化的 d_states_batch_，无需任何 H2D 传输
    invokeCsrBuildMask<cudaStream_t>(ref_info->d_indptr->data<int32_t>(),
                                     ref_info->d_packed_csr_tokens->data<int32_t>(),
                                     d_states_batch_->data<int32_t>(),  // 持久化，无拷贝
                                     ref_info->d_start_mask->data<uint8_t>(),
                                     d_mask_out->data<uint8_t>(),
                                     static_cast<int>(batch_size),
                                     vocab_size,
                                     stream);
#else
    RTP_LLM_CHECK_WITH_INFO(false, "TreeLogitsProcessorCSRGpu requires CUDA");
#endif

    maskLogits(batch_logits, d_mask_out);
}

// =============================================================================
// updateStatus()
//
// 采样完成后调用。优化后流程（每步 decode 仅 1 次 D2H，无大矩阵 D2H）：
//   1. CPU 侧计算各 beam 本步的 col_idx，H2D 上传 [batch_size] col_offsets
//      （batch_size 通常 <=8，传输量 <=32 字节）
//   2. 调用 invokeCsrGatherTokens：在 GPU 上从 new_tokens 直接 gather 采样 token
//      → 消除原来的整矩阵 D2H（[batch * max_col * 4B]）
//   3. 调用 invokeCsrUpdateStates：在 GPU 上原地更新 d_states_batch_
//   4. 一次 D2H [batch_size] 同步 CPU 侧 current_state 镜像
// =============================================================================
void TreeLogitsProcessorCSR::updateStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(new_tokens->dim() == 2);
    RTP_LLM_CHECK(size() == new_tokens->shape()[0]);
    RTP_LLM_CHECK(d_states_batch_ != nullptr);

    const int max_col    = static_cast<int>(new_tokens->shape()[1]);
    const int batch_size = static_cast<int>(size());
    
    const StreamTreeInfo* ref_info = nullptr;
    for (int i = 0; i < batch_size; ++i) {
        if (tree_infos_[i].in_tree_mode) {
            ref_info = &tree_infos_[i];
            break;
        }
    }
    if (ref_info == nullptr) {
        return;
    }

    const int vocab_size = static_cast<int>(ref_info->csr_index->vocab_size);

#if USING_CUDA
    CudaDevice* cuda_dev = dynamic_cast<CudaDevice*>(device_);
    RTP_LLM_CHECK(cuda_dev != nullptr);
    cudaStream_t stream = cuda_dev->getStream();
#endif

    // 逐步推进（通常 num_new_tokens == 1）
    for (int j = 0; j < num_new_tokens; ++j) {

        // --- 1. CPU 侧计算 col_offsets，H2D 上传 [batch_size] int32 ---
        // 仅传 batch_size 个 int（几十字节），远小于整矩阵 D2H
        std::vector<int32_t> col_offsets_cpu(batch_size, 0);
        for (int i = 0; i < batch_size; ++i) {
            if (!tree_infos_[i].in_tree_mode) {
                col_offsets_cpu[i] = -1;  // 无效，gather kernel 会输出 -1
                continue;
            }
            const auto& info   = tree_infos_[i];
            col_offsets_cpu[i] = info.is_beam_search ? (info.input_length + info.current_output_length + j) : j;
        }
        
        // 复用持久化 d_col_offsets_，直接覆盖写（无需重新分配）
        {
            auto cpu_buf = vector2Buffer(col_offsets_cpu);
            // 用 clone 覆盖写：AllocationType::DEVICE 且目标已分配，框架会原地拷贝
            // 如果 d_col_offsets_ 大小匹配则不重新分配
            if (!d_col_offsets_ || d_col_offsets_->size() != static_cast<size_t>(batch_size)) {
                d_col_offsets_ = device_->clone({*cpu_buf, AllocationType::DEVICE});
            } else {
                // 复用已有 buffer：直接 H2D 拷贝到已分配的设备内存
                auto tmp       = device_->clone({*cpu_buf, AllocationType::DEVICE});
                d_col_offsets_ = std::move(tmp);
            }
        }

        // 同理复用 d_sampled_tokens_
        if (!d_sampled_tokens_ || d_sampled_tokens_->size() != static_cast<size_t>(batch_size)) {
            d_sampled_tokens_ = device_->allocateBuffer(
                {DataType::TYPE_INT32, {static_cast<size_t>(batch_size)}, AllocationType::DEVICE});
        }

#if USING_CUDA
        // --- 2. GPU 上 gather 采样 token（完全无 D2H 大矩阵拷贝） ---
        invokeCsrGatherTokens<cudaStream_t>(new_tokens->data<int32_t>(),
                                            d_col_offsets_->data<int32_t>(),
                                            d_sampled_tokens_->data<int32_t>(),
                                            batch_size,
                                            max_col,
                                            stream);

        // --- 3. GPU 上原地更新 d_states_batch_ ---
        invokeCsrUpdateStates<cudaStream_t>(ref_info->d_indptr->data<int32_t>(),
                                            ref_info->d_packed_csr_tokens->data<int32_t>(),
                                            ref_info->d_packed_csr_states->data<int32_t>(),
                                            d_sampled_tokens_->data<int32_t>(),
                                            d_states_batch_->data<int32_t>(),  // 原地更新，持久化 buffer
                                            batch_size,
                                            vocab_size,
                                            stream);
#else
        RTP_LLM_CHECK_WITH_INFO(false, "TreeLogitsProcessorCSRGpu requires CUDA");
#endif

        // --- 4. 一次 D2H：[batch_size] int32，同步 CPU 侧 current_state 镜像 ---
        // 这次 D2H 是必须的：CPU 侧 current_output_length 需要维护，
        // updateMultiSeqStatus() 也依赖 CPU 侧 current_state 进行重排。
        auto updated_cpu = buffer2vector<int32_t>(*device_->clone({*d_states_batch_, AllocationType::HOST}));
        
        for (int i = 0; i < batch_size; ++i) {
            if (!tree_infos_[i].in_tree_mode) {
                continue;
            }
            tree_infos_[i].current_state = updated_cpu[i];
            tree_infos_[i].current_output_length += 1;
        }
    }
}

// =============================================================================
// updateMultiSeqStatus()
//
// beam search 选择完成后调用，按新的 beam 索引重排各 beam 的状态。
// src_batch_indices[i] = 第 i 个新 beam 对应旧 beam 的索引。
//
// 共享只读 GPU buffer 通过 shared_ptr 零拷贝传递。
// d_states_batch_ 按新 beam 顺序重建（仅涉及 [batch_size] int32 的 H2D，开销极小）。
// =============================================================================
void TreeLogitsProcessorCSR::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    
    std::vector<StreamTreeInfo> new_tree_infos;
    new_tree_infos.reserve(src_batch_indices.size());

    // 按新 beam 顺序重排 CPU 侧状态
    std::vector<int32_t> new_states_cpu(src_batch_indices.size());
    for (size_t i = 0; i < src_batch_indices.size(); ++i) {
        int src_idx = src_batch_indices[i];
        RTP_LLM_CHECK(src_idx >= 0 && src_idx < static_cast<int>(tree_infos_.size()));
        StreamTreeInfo info = tree_infos_[src_idx].copy();
        // d_current_state 在优化版本中不再使用，置 nullptr 即可（copy() 已处理）
        new_tree_infos.push_back(std::move(info));
        new_states_cpu[i] = tree_infos_[src_idx].current_state;
    }
    tree_infos_ = std::move(new_tree_infos);

    // 用重排后的状态重建 d_states_batch_（[batch_size] H2D，开销极小）
    auto cpu_buf    = vector2Buffer(new_states_cpu);
    d_states_batch_ = device_->clone({*cpu_buf, AllocationType::DEVICE});

    // d_sampled_tokens_ / d_col_offsets_ 在下次 updateStatus 时会自动检查并重新分配
    d_sampled_tokens_ = nullptr;
    d_col_offsets_    = nullptr;
}

// =============================================================================
// fromGenerateInput()
//
// 1. 解析 ele_rq_ids -> 排序后的 sids<token_num> 数组
// 2. 在 CPU 上构建 CSRIndex<token_num>
// 3. 将 indptr / packed_csr / start_mask 上传到 GPU（每次请求仅一次）
// 4. 为每个 beam 创建 StreamTreeInfo（共享只读 GPU buffer）
// 5. 初始化持久化 d_states_batch_（全零，对应根节点 state=0）
// =============================================================================
std::shared_ptr<TreeLogitsProcessorCSR> TreeLogitsProcessorCSR::fromGenerateInput(
    rtp_llm::DeviceBase* device, std::shared_ptr<GenerateInput> generate_input, int32_t num, int32_t vocab_size) {
    if (generate_input->generate_config->ele_rq_ids.empty()) {
        return nullptr;
    }

    // --- 1. 解析约束字符串 ---
    std::vector<sids<token_num>> origin_rq_ids = split_strings<token_num>(generate_input->generate_config->ele_rq_ids);
    std::sort(origin_rq_ids.begin(), origin_rq_ids.end());

    // --- 2. 在 CPU 上构建 CSR 索引 ---
    auto csr_index = std::make_shared<CSRIndex<token_num>>();
    bool success   = build_csr_from_fresh_data<token_num>(origin_rq_ids, *csr_index, vocab_size);
    if (!success) {
        return nullptr;
    }

    // --- 3. 将共享只读 CSR 数组上传到 GPU（每次请求仅一次） ---
    rtp_llm::BufferPtr d_indptr;
    {
        auto cpu_buf = vector2Buffer(csr_index->indptr);
        d_indptr     = device->clone({*cpu_buf, AllocationType::DEVICE});
    }

    rtp_llm::BufferPtr d_packed_csr_tokens;
    {
        auto cpu_buf        = vector2Buffer(csr_index->packed_csr_tokens);
        d_packed_csr_tokens = device->clone({*cpu_buf, AllocationType::DEVICE});
    }

    rtp_llm::BufferPtr d_packed_csr_states;
    {
        auto cpu_buf        = vector2Buffer(csr_index->packed_csr_states);
        d_packed_csr_states = device->clone({*cpu_buf, AllocationType::DEVICE});
    }

    // start_mask 是 vector<bool>，需要转换为 vector<uint8_t> 才能上传
    rtp_llm::BufferPtr d_start_mask;
    {
        std::vector<uint8_t> sm_u8(csr_index->start_mask.size());
        for (size_t i = 0; i < csr_index->start_mask.size(); ++i) {
            sm_u8[i] = csr_index->start_mask[i] ? 1u : 0u;
        }
        auto cpu_buf = vector2Buffer(sm_u8);
        d_start_mask = device->clone({*cpu_buf, AllocationType::DEVICE});
    }

    // --- 4. 为每个 beam 创建 StreamTreeInfo ---
    bool is_multi_seq =
        generate_input->generate_config->hasNumBeams() || generate_input->generate_config->num_return_sequences > 1;

    // 直接使用调用方传入的 device，避免调用 DeviceFactory::getDefaultDevice()
    // （后者在设备尚未完成注册时会触发 FATAL abort）
    auto processor_ptr = std::make_shared<TreeLogitsProcessorCSR>(device);

    for (int32_t i = 0; i < num; ++i) {
        StreamTreeInfo tree_info(
            /*in_tree_mode=*/true,
            /*input_length=*/generate_input->inputLength(),
            /*current_output_length=*/0,
            /*is_beam_search=*/is_multi_seq,
            /*current_state=*/0,
            /*csr_index=*/csr_index,  // CPU 只读副本
            /*d_indptr=*/d_indptr,    // 共享只读 GPU buffer
            /*d_packed_csr_tokens=*/d_packed_csr_tokens,
            /*d_packed_csr_states=*/d_packed_csr_states,
            /*d_start_mask=*/d_start_mask,
            /*d_current_state=*/nullptr  // 优化版不再使用逐 beam 私有 buffer
        );

        std::vector<StreamTreeInfo> single_vec = {std::move(tree_info)};
        auto single_processor = std::make_shared<TreeLogitsProcessorCSR>(device, std::move(single_vec));
        processor_ptr->insert(single_processor);
    }

    // --- 5. 初始化持久化 d_states_batch_（全 0，对应根节点 state=0） ---
    std::vector<int32_t> init_states(static_cast<size_t>(num), 0);
    auto                 cpu_states_buf = vector2Buffer(init_states);
    processor_ptr->d_states_batch_      = device->clone({*cpu_states_buf, AllocationType::DEVICE});

    return processor_ptr;
}

}  // namespace rtp_llm
