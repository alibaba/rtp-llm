#include <cuda_runtime.h>
#include <pccl_device.h>
#include <cstdint>

struct Context {
    pcclComm_t comm;
    pcclDevComm dev;
    pcclWindow_t txwin, rxwin;
    uint32_t *tx, *rx;
    uint64_t* sequence;
    size_t stride, row_bytes;
    int rank, size;
};

// Dispatch and combine use different windows. The opposite exchange proves
// that peers consumed this window before its next use, without a ready barrier.
__global__ void begin_exchange(pcclDevComm dev, pcclWindow_t txwin, pcclWindow_t rxwin,
                               uint32_t* tx, uint32_t* rx, uint64_t* sequence,
                               size_t stride, size_t row_bytes, int rank, int size) {
    const int peer = blockIdx.x;
    if (threadIdx.x == 0 && peer == 0) ++*sequence;
    if (threadIdx.x == 0 && peer != rank) {
        pcclGin gin(dev, peer);
        const auto team = pcclTeamWorld(dev);
        const size_t bytes = 16 + tx[peer * stride / 4] * row_bytes;
        gin.put(team, peer, rxwin, rank * stride, txwin, peer * stride, bytes,
                pcclGin_SignalInc{0}, pcclGin_CounterInc{0});
    }
    const size_t words = (16 + tx[rank * stride / 4] * row_bytes) / 4;
    if (peer == rank)
        for (size_t i = threadIdx.x; i < words; i += blockDim.x)
            rx[rank * stride / 4 + i] = tx[rank * stride / 4 + i];
}

__global__ void finish_exchange(pcclDevComm dev, uint64_t* sequence, int rank, int size) {
    if (threadIdx.x == 0) {
        const int peer = blockIdx.x;
        pcclGin gin(dev, peer);
        if (peer == rank)
            gin.waitSignal(pcclCoopThread{}, 0, *sequence * (size - 1));
        else
            gin.waitCounter(pcclCoopThread{}, 0, *sequence);
    }
}

static int create_context(void* comm, void* tx, void* rx, size_t stride,
                          size_t row_bytes, int rank, int size,
                          size_t sequence_count, void** result) {
    *result = nullptr;
    auto* ctx = new Context{};
    ctx->comm = static_cast<pcclComm_t>(comm);
    ctx->tx = static_cast<uint32_t*>(tx);
    ctx->rx = static_cast<uint32_t*>(rx);
    ctx->stride = stride; ctx->row_bytes = row_bytes;
    ctx->rank = rank; ctx->size = size;
    bool tx_registered = false, rx_registered = false, dev_created = false;
    auto fail = [&](int error) {
        if (ctx->sequence) cudaFree(ctx->sequence);
        if (dev_created) pcclDevCommDestroy(ctx->comm, &ctx->dev);
        if (rx_registered) pcclCommWindowDeregister(ctx->comm, ctx->rxwin);
        if (tx_registered) pcclCommWindowDeregister(ctx->comm, ctx->txwin);
        delete ctx;
        return error;
    };
    auto rc = pcclCommWindowRegister(ctx->comm, tx, stride * size,
                                     &ctx->txwin, PCCL_WIN_COLL_SYMMETRIC);
    if (rc) return fail(100 + rc);
    tx_registered = true;
    rc = pcclCommWindowRegister(ctx->comm, rx, stride * size,
                                &ctx->rxwin, PCCL_WIN_COLL_SYMMETRIC);
    if (rc) return fail(200 + rc);
    rx_registered = true;
    pcclDevCommRequirements req = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    req.ginForceEnable = true;
    req.ginConnectionType = PCCL_GIN_CONNECTION_FULL;
    req.ginContextCount = size; req.ginSignalCount = 1; req.ginCounterCount = 1;
    rc = pcclDevCommCreate(ctx->comm, &req, &ctx->dev);
    if (rc) return fail(300 + rc);
    dev_created = true;
    auto gpu_rc = cudaMalloc(reinterpret_cast<void**>(&ctx->sequence), sequence_count * sizeof(uint64_t));
    if (gpu_rc) return fail(400 + gpu_rc);
    gpu_rc = cudaMemset(ctx->sequence, 0, sequence_count * sizeof(uint64_t));
    if (gpu_rc) return fail(500 + gpu_rc);
    *result = ctx;
    return 0;
}

extern "C" int sparse_gin_create(void* comm, void* tx, void* rx, size_t stride,
                                 size_t row_bytes, int rank, int size, void** result) {
    return create_context(comm, tx, rx, stride, row_bytes, rank, size, 1, result);
}

extern "C" int sparse_gin_begin(void* handle, void* stream) {
    auto* c = static_cast<Context*>(handle);
    begin_exchange<<<c->size, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        c->dev, c->txwin, c->rxwin, c->tx, c->rx, c->sequence,
        c->stride, c->row_bytes, c->rank, c->size);
    return cudaGetLastError();
}

extern "C" int sparse_gin_finish(void* handle, void* stream) {
    auto* c = static_cast<Context*>(handle);
    finish_exchange<<<c->size, 32, 0, static_cast<cudaStream_t>(stream)>>>(c->dev, c->sequence, c->rank, c->size);
    return cudaGetLastError();
}

extern "C" int sparse_gin_destroy(void* handle) {
    auto* c = static_cast<Context*>(handle);
    auto gpu_rc = cudaDeviceSynchronize();
    if (gpu_rc) return 500 + gpu_rc;
    auto rc = pcclDevCommDestroy(c->comm, &c->dev);
    if (rc) return rc;
    pcclCommWindowDeregister(c->comm, c->txwin);
    pcclCommWindowDeregister(c->comm, c->rxwin);
    cudaFree(c->sequence);
    delete c;
    return 0;
}
