#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

using namespace std;

namespace rtp_llm {

ReduceScatterLoraLinearOutput CudaDevice::loraLinearReduceScatter(const LoraLinearReduceScatterParams& params) {
    size_t overlap_comm_type = init_params_.device_resource_config.overlap_comm_type;
    if (overlap_comm_type <= 1) {
        return DeviceBase::loraLinearReduceScatter(params);
    }
    const LoraLinearParams& linear_params = params.lora_linear_params;
    const auto&             gemm_a        = linear_params.gemm_params.A;
    const auto&             gemm_b        = linear_params.gemm_params.B;
    const size_t            m             = gemm_a.shape()[0];
    const size_t            n             = gemm_b.shape()[1];
    size_t   tp_size          = params.mode == ParallelMode::FFN_TP ? init_params_.ffn_tp_size : init_params_.tp_size;
    size_t   tp_rank          = params.mode == ParallelMode::FFN_TP ? init_params_.ffn_tp_rank : init_params_.tp_rank;
    DataType output_data_type = params.output_type;

    if ((overlap_comm_type == 2) && ((!linear_params.lora_input) || linear_params.lora_input->isEmpty())
        && init_params_.tp_size > 1) {
        CommBuffer* cb = nullptr;
        // Get communication and GEMM input chunk sizes
        const auto m_chunk = m / tp_size;

        if (params.mode == ParallelMode::TP) {
            cb = attn_rs_comm_buffer_.get();
        } else if (params.mode == ParallelMode::FFN_TP) {
            cb = ffn_rs_comm_buffer_.get();
        } else {
            RTP_LLM_CHECK("unavailable");
        }
        Communicator* comm = cb->_comm;

        RTP_LLM_LOG_DEBUG("m %d, n %d", m, n);
        BufferPtr output_d = BufferPtr(new Buffer(MemoryType::MEMORY_GPU, output_data_type, {m, n}, cb->_ubuf));

        // Get communication and GEMM output chunk sizes
        const int comm_bytes = output_d->sizeBytes() / tp_size;

        if (cudaGetLastError() != cudaSuccess) {
            RTP_LLM_LOG_INFO(cudaGetErrorString(cudaGetLastError()));
        }

        check_cuda_value(cudaEventRecord(cb->_start_compute, stream_));
        // Catch up the main stream
        for (size_t i = 0; i < cb->_stream_send.size(); i++) {
            check_cuda_value(cudaStreamWaitEvent(cb->_stream_send[i], cb->_start_compute, 0));
        }
        check_cuda_value(cudaStreamWaitEvent(cb->_stream_recv, cb->_start_compute, 0));
        for (size_t i = 0; i < cb->_stream_compute.size(); i++) {
            check_cuda_value(cudaStreamWaitEvent(cb->_stream_compute[i], cb->_start_compute, 0));
        }

        // GEMM and send/recv chunks
        for (int i = 0; i < tp_size; i++) {
            // GEMM chunk
            int input_a_chunk_id = (tp_rank + i + 1) % tp_size;

            BufferPtr input_a_chunk = nullptr;
            if (params.qscheme == NoQuantize) {
                input_a_chunk = gemm_a.slice(input_a_chunk_id * m_chunk, m_chunk);
            } else if (params.qscheme == Qint8PerToken) {
                input_a_chunk = reinterpret_cast<const QBuffer&>(gemm_a).qslice(input_a_chunk_id * m_chunk, m_chunk);
            } else if (params.qscheme == Qfp8PerTensor) {
                input_a_chunk =
                    reinterpret_cast<const QBuffer&>(gemm_a).qslicePerTensor(input_a_chunk_id * m_chunk, m_chunk);
            } else {
                RTP_LLM_FAIL("unsupported qscheme");
            }
            BufferPtr output_chunk = output_d->slice(i * m_chunk, m_chunk);

            auto gemm_params = GemmParams(*input_a_chunk,
                                          linear_params.gemm_params.B,
                                          linear_params.gemm_params.C,
                                          output_chunk,
                                          linear_params.gemm_params.compute_type,
                                          linear_params.gemm_params.D_type,
                                          linear_params.gemm_params.transA,
                                          linear_params.gemm_params.transB,
                                          linear_params.gemm_params.activationType,
                                          linear_params.gemm_params.alpha,
                                          linear_params.gemm_params.beta,
                                          init_params_.device_resource_config.overlap_math_sm_count,
                                          cb->_stream_compute[i % cb->_stream_compute.size()]);
            loraLinear({gemm_params});
            printBufferData(*input_a_chunk, "input_a_chunk");
            printBufferData(*output_chunk, "output_chunk");

            if (i < tp_size - 1) {
                int cur_stream_id = i % cb->_stream_compute.size();
                int send_offset   = comm_bytes * i;
                int recv_offset   = comm_bytes * (i + tp_size);
                int send_rank     = (tp_rank + i + 1) % tp_size;
                int recv_rank     = (tp_size + tp_rank - i - 1) % tp_size;
                check_cuda_value(cudaEventRecord(cb->_start_comm, cb->_stream_compute[cur_stream_id]));
                check_cuda_value(cudaStreamWaitEvent(cb->_stream_send[cur_stream_id], cb->_start_comm, 0));
                check_cuda_value(cudaStreamWaitEvent(cb->_stream_recv, cb->_start_comm, 0));
                RTP_LLM_LOG_DEBUG("comm_bytes %d, output_d->sizeBytes() %d, output_data_type %d i %d send_rank %d",
                                  comm_bytes,
                                  output_d->sizeBytes(),
                                  output_data_type,
                                  i,
                                  send_rank);
                userbuffers_send(cb->_ub_reg,
                                 send_offset,
                                 recv_offset,
                                 comm_bytes,
                                 comm,
                                 send_rank,
                                 cb->_stream_send[cur_stream_id]);
                userbuffers_recv(cb->_ub_reg, comm, recv_rank, cb->_stream_recv);
            }
        }

        for (size_t i = 0; i < cb->_stream_compute.size(); i++) {
            check_cuda_value(cudaEventRecord(cb->_stop_compute, cb->_stream_compute[i]));
            check_cuda_value(cudaStreamWaitEvent(stream_, cb->_stop_compute, 0));
        }
        for (size_t i = 0; i < cb->_stream_send.size(); i++) {
            check_cuda_value(cudaEventRecord(cb->_stop_send, cb->_stream_send[i]));
            check_cuda_value(cudaStreamWaitEvent(stream_, cb->_stop_send, 0));
        }
        check_cuda_value(cudaEventRecord(cb->_stop_recv, cb->_stream_recv));
        check_cuda_value(cudaStreamWaitEvent(stream_, cb->_stop_recv, 0));

        // Reduce GEMM output chunks
        RTP_LLM_CHECK_WITH_INFO(output_d->type() == DataType::TYPE_FP16 || output_d->type() == DataType::TYPE_BF16,
                                "ReduceScatter only supports BF16/FP16 output.");
        char*     reduce_buf_ptr = reinterpret_cast<char*>(cb->_ubuf) + (tp_size - 1) * comm_bytes;
        char*     rs_output_ptr  = reinterpret_cast<char*>(params.rs_recv_buffer->data());
        BufferPtr gemm_output    = BufferPtr(new Buffer(MemoryType::MEMORY_GPU,
                                                     output_data_type,
                                                        {m, n},
                                                     reinterpret_cast<char*>(cb->_ubuf) + (tp_size - 1) * comm_bytes));
        invokeLocalReduceDispatch(
            output_data_type, reduce_buf_ptr, rs_output_ptr, tp_size, output_d->size() / tp_size, stream_);
        return ReduceScatterLoraLinearOutput({std::move(params.rs_recv_buffer), std::move(gemm_output)});
    }
    return DeviceBase::loraLinearReduceScatter(params);
}

AllGatherLoraLinearOutput CudaDevice::allGatherloraLinear(const AllGatherLoraLinearParams& params) {
    size_t overlap_comm_type = init_params_.device_resource_config.overlap_comm_type;
    if (overlap_comm_type <= 1) {
        return DeviceBase::allGatherloraLinear(params);
    }
    const LoraLinearParams& linear_params = params.lora_linear_params;
    const auto&             gemm_a        = linear_params.gemm_params.A;
    const auto&             gemm_b        = linear_params.gemm_params.B;
    const size_t            m             = gemm_a.shape()[0];
    const size_t            k             = gemm_a.shape()[1];
    const size_t            n             = gemm_b.shape()[1];

    BufferPtr output = nullptr;
    if (linear_params.gemm_params.D) {
        output = linear_params.gemm_params.D;
    } else {
        output                      = allocateBuffer({params.output_type, {m, n}});
        linear_params.gemm_params.D = output;
    }
    size_t tp_size = params.mode == ParallelMode::FFN_TP ? init_params_.ffn_tp_size : init_params_.tp_size;
    size_t tp_rank = params.mode == ParallelMode::FFN_TP ? init_params_.ffn_tp_rank : init_params_.tp_rank;

    if ((overlap_comm_type == 2) && ((!linear_params.lora_input) || linear_params.lora_input->isEmpty())
        && init_params_.tp_size > 1) {
        CommBuffer* cb       = nullptr;
        CommBuffer* scale_cb = nullptr;

        const auto m_chunk = m / tp_size;
        if (params.mode == ParallelMode::TP) {
            cb = attn_ag_comm_buffer_.get();
            if (params.qscheme == Qint8PerToken) {
                scale_cb = attn_ag_scale_comm_buffer_.get();
            }
        } else if (params.mode == ParallelMode::FFN_TP) {
            cb = ffn_ag_comm_buffer_.get();
            if (params.qscheme == Qint8PerToken) {
                scale_cb = ffn_ag_scale_comm_buffer_.get();
            }
        } else {
            RTP_LLM_CHECK("unavailable");
        }
        Communicator* comm = cb->_comm;

        // Get communication and GEMM output chunk sizes
        const int comm_bytes = params.ag_send_buffer->sizeBytes();

        int scale_comm_bytes = 0;
        if (params.qscheme == Qint8PerToken) {
            scale_comm_bytes = reinterpret_cast<const QBuffer&>(*params.ag_send_buffer).scales().sizeBytes();
        }

        check_cuda_value(cudaEventRecord(cb->_start_compute, stream_));
        check_cuda_value(cudaStreamWaitEvent(cb->_stream_send[0], cb->_start_compute, 0));
        check_cuda_value(cudaStreamWaitEvent(cb->_stream_recv, cb->_start_compute, 0));
        for (size_t i = 0; i < cb->_stream_compute.size(); i++) {
            check_cuda_value(cudaStreamWaitEvent(cb->_stream_compute[i], cb->_start_compute, 0));
        }

        BufferPtr gemm_a = nullptr;
        if (params.qscheme == NoQuantize) {
            gemm_a = BufferPtr(new Buffer(MemoryType::MEMORY_GPU, params.ag_send_buffer->type(), {m, k}, cb->_ubuf));
        } else if (params.qscheme == Qint8PerToken) {
            BufferPtr gemm_a_kernel =
                BufferPtr(new Buffer(MemoryType::MEMORY_GPU, params.ag_send_buffer->type(), {m, k}, cb->_ubuf));
            BufferPtr gemm_a_sacle =
                BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_FP32, {m}, scale_cb->_ubuf));
            gemm_a = BufferPtr(new QBuffer(
                std::move(gemm_a_kernel),
                std::move(gemm_a_sacle),
                std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
        } else if (params.qscheme == Qfp8PerTensor) {
            BufferPtr gemm_a_kernel =
                BufferPtr(new Buffer(MemoryType::MEMORY_GPU, params.ag_send_buffer->type(), {m, k}, cb->_ubuf));
            BufferPtr gemm_a_sacle =
                BufferPtr(new Buffer(MemoryType::MEMORY_GPU,
                                     DataType::TYPE_FP32,
                                     {1},
                                     reinterpret_cast<const QBuffer&>(linear_params.gemm_params.A).scalesData()));
            gemm_a = BufferPtr(new QBuffer(
                std::move(gemm_a_kernel),
                std::move(gemm_a_sacle),
                std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
        } else {
            RTP_LLM_FAIL("unsupported qscheme");
        }

        for (int i = 0; i < tp_size; i++) {
            int send_chunk_id = (tp_size + tp_rank - i) % tp_size;
            int send_offset   = comm_bytes * send_chunk_id;

            BufferPtr input_a_chunk = nullptr;
            if (params.qscheme == NoQuantize) {
                input_a_chunk = gemm_a->slice(send_chunk_id * m_chunk, m_chunk);
            } else if (params.qscheme == Qint8PerToken) {
                input_a_chunk =
                    reinterpret_cast<const QBuffer*>(gemm_a.get())->qslice(send_chunk_id * m_chunk, m_chunk);
            } else if (params.qscheme == Qfp8PerTensor) {
                input_a_chunk =
                    reinterpret_cast<const QBuffer*>(gemm_a.get())->qslicePerTensor(send_chunk_id * m_chunk, m_chunk);
            } else {
                RTP_LLM_FAIL("unsupported qscheme");
            }
            BufferPtr output_chunk = output->slice(send_chunk_id * m_chunk, m_chunk);

            auto gemm_params = GemmParams(*input_a_chunk,
                                          linear_params.gemm_params.B,
                                          linear_params.gemm_params.C,
                                          output_chunk,
                                          linear_params.gemm_params.compute_type,
                                          linear_params.gemm_params.D_type,
                                          linear_params.gemm_params.transA,
                                          linear_params.gemm_params.transB,
                                          linear_params.gemm_params.activationType,
                                          linear_params.gemm_params.alpha,
                                          linear_params.gemm_params.beta,
                                          init_params_.device_resource_config.overlap_math_sm_count,
                                          cb->_stream_compute[i % cb->_stream_compute.size()]);
            loraLinear({gemm_params});

            if (i < tp_size - 1) {
                userbuffers_send(
                    cb->_ub_reg, send_offset, send_offset, comm_bytes, comm, cb->_next_rank, cb->_stream_send[0]);
                userbuffers_recv(cb->_ub_reg, comm, cb->_prev_rank, cb->_stream_recv);

                if (params.qscheme == Qint8PerToken) {
                    userbuffers_send(scale_cb->_ub_reg,
                                     scale_comm_bytes * send_chunk_id,
                                     scale_comm_bytes * send_chunk_id,
                                     scale_comm_bytes,
                                     scale_cb->_comm,
                                     cb->_next_rank,
                                     cb->_stream_send[0]);
                    userbuffers_recv(scale_cb->_ub_reg, scale_cb->_comm, cb->_prev_rank, cb->_stream_recv);
                }
                check_cuda_value(cudaEventRecord(cb->_stop_recv, cb->_stream_recv));
                check_cuda_value(cudaStreamWaitEvent(cb->_stream_send[0], cb->_stop_recv, 0));
                check_cuda_value(
                    cudaStreamWaitEvent(cb->_stream_compute[(i + 1) % cb->_stream_compute.size()], cb->_stop_recv, 0));
            }
        }

        for (size_t i = 0; i < cb->_stream_compute.size(); i++) {
            check_cuda_value(cudaEventRecord(cb->_stop_compute, cb->_stream_compute[i]));
            check_cuda_value(cudaStreamWaitEvent(stream_, cb->_stop_compute, 0));
        }
        check_cuda_value(cudaEventRecord(cb->_stop_send, cb->_stream_send[0]));
        check_cuda_value(cudaStreamWaitEvent(stream_, cb->_stop_send, 0));
        check_cuda_value(cudaEventRecord(cb->_stop_recv, cb->_stream_recv));
        check_cuda_value(cudaStreamWaitEvent(stream_, cb->_stop_recv, 0));

        return AllGatherLoraLinearOutput({std::move(output), std::move(gemm_a)});
    }
    return DeviceBase::allGatherloraLinear(params);
}

}  // namespace rtp_llm