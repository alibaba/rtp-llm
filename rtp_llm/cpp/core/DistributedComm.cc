#include "rtp_llm/cpp/core/DistributedComm.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#elif USING_ROCM
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#endif

#if USING_CUDA
using DeviceGuard = at::cuda::CUDAGuard;
#elif USING_ROCM
using DeviceGuard = c10::hip::HIPGuardMasqueradingAsCUDA;
#endif

namespace rtp_llm {

// ============================================================
// Global PgMapState — single definition (non-inline)
// ============================================================

namespace detail {
PgMapState& globalPgState() {
    static PgMapState instance;
    return instance;
}
}  // namespace detail

namespace {

using detail::ProcessGroupEntry;

static c10d::ReduceOp::RedOpType toC10dReduceOp(ReduceOp op) {
    switch (op) {
        case ReduceOp::Sum:
            return c10d::ReduceOp::SUM;
        case ReduceOp::Prod:
            return c10d::ReduceOp::PRODUCT;
        case ReduceOp::Max:
            return c10d::ReduceOp::MAX;
        case ReduceOp::Min:
            return c10d::ReduceOp::MIN;
        case ReduceOp::Avg:
            return c10d::ReduceOp::AVG;
        default:
            RTP_LLM_LOG_ERROR("Unknown ReduceOp: %d", static_cast<int>(op));
            return c10d::ReduceOp::SUM;
    }
}

static ProcessGroupEntry getEntry(ParallelMode mode) {
    auto&                       state = detail::globalPgState();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto                        it = state.map.find(mode);
    if (it == state.map.end()) {
        RTP_LLM_LOG_ERROR("No ProcessGroup registered for ParallelMode %d", static_cast<int>(mode));
        throw std::runtime_error("ProcessGroup not registered for requested ParallelMode");
    }
    return it->second;
}

}  // anonymous namespace

void registerProcessGroup(ParallelMode mode, c10::intrusive_ptr<c10d::ProcessGroup> pg, int device_id) {
    auto&                       state = detail::globalPgState();
    std::lock_guard<std::mutex> lock(state.mutex);
    ProcessGroupEntry           entry;
    entry.pg         = std::move(pg);
    entry.rank       = entry.pg->getRank();
    entry.world_size = entry.pg->getSize();
    entry.device_id  = device_id;
    state.map[mode]  = std::move(entry);
}

c10::intrusive_ptr<c10d::ProcessGroup> getProcessGroup(ParallelMode mode) {
    return getEntry(mode).pg;
}

bool hasProcessGroup(ParallelMode mode) {
    auto&                       state = detail::globalPgState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.map.count(mode) > 0;
}

void clearProcessGroups() {
    auto&                       state = detail::globalPgState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.map.clear();
}

// ============================================================
// c10d Communication Ops
// NCCL requires CUDA tensors on the correct device for each rank.
// We use CUDAGuard to pin the device, and transparently move
// CPU tensors to the rank's GPU when needed.
// ============================================================

static at::Tensor ensureCuda(const at::Tensor& t, int device_id) {
    if (t.is_cuda())
        return t;
    return t.to(at::Device(at::kCUDA, device_id));
}

void c10dBroadcast(const BroadcastParams& params) {
    if (!hasProcessGroup(params.mode)) {
        return;
    }
    auto entry = getEntry(params.mode);
    if (entry.world_size < 2) {
        return;
    }

    DeviceGuard guard(entry.device_id);
    for (auto& buffer : params.buffers) {
        bool                    on_cpu  = !buffer.is_cuda();
        at::Tensor              gpu_buf = on_cpu ? buffer.to(at::Device(at::kCUDA, entry.device_id), true) : buffer;
        std::vector<at::Tensor> tensors = {gpu_buf};
        c10d::BroadcastOptions  opts;
        opts.rootRank = params.root;
        auto work     = entry.pg->broadcast(tensors, opts);
        work->wait();
        if (on_cpu) {
            buffer.copy_(tensors[0]);
        }
    }
}

AllReduceOutput c10dAllReduce(const AllReduceParams& params) {
    if (!hasProcessGroup(params.mode)) {
        return AllReduceOutput{params.buffer};
    }
    auto entry = getEntry(params.mode);
    if (entry.world_size < 2) {
        return AllReduceOutput{params.buffer};
    }

    DeviceGuard guard(entry.device_id);

    auto&       buffer      = params.buffer;
    const auto& dest_buffer = params.dest.defined() ? params.dest : buffer;

    if (params.dest.defined()) {
        dest_buffer.copy_(buffer);
    }

    bool                    on_cpu  = !dest_buffer.is_cuda();
    at::Tensor              gpu_buf = on_cpu ? dest_buffer.to(at::Device(at::kCUDA, entry.device_id)) : dest_buffer;
    std::vector<at::Tensor> tensors = {gpu_buf};
    c10d::AllreduceOptions  opts;
    opts.reduceOp = toC10dReduceOp(params.op);
    auto work     = entry.pg->allreduce(tensors, opts);
    work->wait();
    if (on_cpu) {
        dest_buffer.copy_(tensors[0]);
    }
    return AllReduceOutput{dest_buffer};
}

void c10dAllGather(const AllGatherParams& params) {
    if (!hasProcessGroup(params.mode)) {
        return;
    }
    auto entry = getEntry(params.mode);
    if (entry.world_size < 2) {
        return;
    }

    DeviceGuard guard(entry.device_id);
    for (size_t i = 0; i < params.recv_buffers.size(); ++i) {
        auto&        recv_buffer = params.recv_buffers[i];
        const size_t data_num    = recv_buffer.numel() / static_cast<size_t>(entry.world_size);
        RUNTIME_ASSERT_OP_ARG(data_num * static_cast<size_t>(entry.world_size)
                                  == static_cast<size_t>(recv_buffer.numel()),
                              "Buffer size %zu must be divisible by world size %d",
                              static_cast<size_t>(recv_buffer.numel()),
                              entry.world_size);

        bool       recv_on_cpu = !recv_buffer.is_cuda();
        at::Tensor gpu_recv    = recv_on_cpu ? recv_buffer.to(at::Device(at::kCUDA, entry.device_id)) : recv_buffer;

        auto gpu_recv_flat = gpu_recv.reshape({-1});

        at::Tensor send_tensor;
        if (params.inplace) {
            send_tensor = gpu_recv_flat.narrow(0, entry.rank * data_num, data_num).contiguous();
        } else {
            send_tensor = ensureCuda(params.send_buffers[i], entry.device_id);
        }

        auto work = entry.pg->_allgather_base(gpu_recv_flat, send_tensor);
        work->wait();
        if (recv_on_cpu) {
            recv_buffer.copy_(gpu_recv);
        }
    }
}

void c10dSyncCommunication(bool timeout) {
    (void)timeout;
}

void c10dSyncCommunication(ParallelMode mode, bool timeout) {
    (void)mode;
    (void)timeout;
}

}  // namespace rtp_llm
