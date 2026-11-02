#include "ppu_sparse_gin.cu"

// Paired serial phases only: the reverse phase proves the peer consumed this
// phase before its tx is reused. Independent exchanges/multiple streams cannot
// use this protocol. Every peer sends a header even when its row count is zero.
__global__ void signal_exchange(pcclDevComm dev, pcclWindow_t txwin,
                                pcclWindow_t rxwin, uint32_t* tx, uint32_t* rx,
                                uint64_t* sequences, size_t stride,
                                size_t row_bytes, int rank, int size) {
    const int peer = blockIdx.x;
    const size_t words = (16 + tx[rank * stride / 4] * row_bytes) / 4;
    if (peer == rank)
        for (size_t i = threadIdx.x; i < words; i += blockDim.x)
            rx[rank * stride / 4 + i] = tx[rank * stride / 4 + i];
    if (threadIdx.x == 0) {
        pcclGin gin(dev, peer);
        if (peer == rank) {
            const uint64_t sequence = ++sequences[peer];
            gin.waitSignal(pcclCoopThread{}, 0, sequence * (size - 1));
        } else {
            const size_t bytes = 16 + tx[peer * stride / 4] * row_bytes;
            gin.put(pcclTeamWorld(dev), peer, rxwin, rank * stride,
                    txwin, peer * stride, bytes, pcclGin_SignalInc{0}, pcclGin_None{});
        }
    }
}

extern "C" int signal_gin_create(void* comm, void* tx, void* rx, size_t stride,
                                 size_t row_bytes, int rank, int size, void** result) {
    return create_context(comm, tx, rx, stride, row_bytes, rank, size, size, result);
}

extern "C" int signal_gin_exchange(void* handle, void* stream) {
    auto* c = static_cast<Context*>(handle);
    signal_exchange<<<c->size, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        c->dev, c->txwin, c->rxwin, c->tx, c->rx, c->sequence,
        c->stride, c->row_bytes, c->rank, c->size);
    return cudaGetLastError();
}

extern "C" int signal_gin_destroy(void* handle) {
    auto* c = static_cast<Context*>(handle);
    auto gpu_rc = cudaDeviceSynchronize();
    if (gpu_rc) return 900 + gpu_rc;
    // There may be no next dispatch to acknowledge the last return. Every peer
    // enters this collective only after its last receive/consume completed.
    // Keep all registered buffers alive if synchronization fails.
    auto rc = pcclAllReduce(c->sequence, c->sequence, 1, pcclUint64, pcclSum,
                            c->comm, nullptr);
    if (rc) return 1000 + rc;
    gpu_rc = cudaDeviceSynchronize();
    if (gpu_rc) return 1100 + gpu_rc;
    return sparse_gin_destroy(handle);
}
