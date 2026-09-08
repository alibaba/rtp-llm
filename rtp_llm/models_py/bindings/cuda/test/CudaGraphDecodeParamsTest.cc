#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <algorithm>
#include <limits>
#include <vector>

#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

namespace rtp_llm {
namespace {

struct DecodeBuffers {
    torch::Tensor lengths, blocks, batches, pages, indptr, last, qo, kvlen, positions, slots;
    int           capacity;
    int           columns;

    DecodeBuffers(int capacity_, int columns_): capacity(capacity_), columns(columns_) {
        const auto opts   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        auto       output = [&](int64_t size) { return torch::full({size}, -777, opts); };
        lengths           = output(capacity);
        blocks            = (torch::arange(static_cast<int64_t>(capacity) * columns, opts).remainder(100003) + 20000000)
                     .reshape({capacity, columns});
        batches   = output(capacity);
        pages     = output(static_cast<int64_t>(capacity) * columns);
        indptr    = output(capacity + 1);
        last      = output(capacity);
        qo        = output(capacity + 1);
        kvlen     = output(capacity);
        positions = output(capacity);
        slots     = torch::full({capacity}, -777, opts.dtype(torch::kInt64));
    }

    void invoke(int batch, int page_size, cudaStream_t stream) {
        invokePrepareFlashInferDecodeParams(lengths.data_ptr<int32_t>(),
                                            blocks.data_ptr<int32_t>(),
                                            batches.data_ptr<int32_t>(),
                                            pages.data_ptr<int32_t>(),
                                            indptr.data_ptr<int32_t>(),
                                            last.data_ptr<int32_t>(),
                                            qo.data_ptr<int32_t>(),
                                            kvlen.data_ptr<int32_t>(),
                                            positions.data_ptr<int32_t>(),
                                            slots.data_ptr<int64_t>(),
                                            batch,
                                            columns,
                                            page_size,
                                            capacity,
                                            stream);
    }

    void check(int batch, int page_size) {
        const auto  seq_h       = lengths.cpu();
        const auto  blocks_h    = blocks.cpu();
        const auto  batches_h   = batches.cpu();
        const auto  pages_h     = pages.cpu();
        const auto  indptr_h    = indptr.cpu();
        const auto  last_h      = last.cpu();
        const auto  qo_h        = qo.cpu();
        const auto  kvlen_h     = kvlen.cpu();
        const auto  positions_h = positions.cpu();
        const auto  slots_h     = slots.cpu();
        const auto* seq         = seq_h.data_ptr<int32_t>();
        const auto* block       = blocks_h.data_ptr<int32_t>();
        const int   ps          = std::max(page_size, 1);
        int         offset      = 0;
        ASSERT_EQ(indptr_h.data_ptr<int32_t>()[0], 0);
        ASSERT_EQ(qo_h.data_ptr<int32_t>()[0], 0);
        for (int b = 0; b < capacity; ++b) {
            SCOPED_TRACE(b);
            if (b < batch) {
                const int64_t length      = std::max(seq[b], 1);
                const int     count       = std::min<int64_t>((length - 1) / ps + 1, columns);
                const int64_t block_index = (length - 1) / ps;
                const int64_t number =
                    block_index < columns ? block[static_cast<int64_t>(b) * columns + block_index] : 0;
                ASSERT_EQ(batches_h.data_ptr<int32_t>()[b], b);
                ASSERT_EQ(kvlen_h.data_ptr<int32_t>()[b], length);
                ASSERT_EQ(positions_h.data_ptr<int32_t>()[b], length - 1);
                ASSERT_EQ(last_h.data_ptr<int32_t>()[b], (length - 1) % ps + 1);
                ASSERT_EQ(slots_h.data_ptr<int64_t>()[b], number * ps + (length - 1) % ps);
                ASSERT_EQ(qo_h.data_ptr<int32_t>()[b + 1], b + 1);
                for (int page = 0; page < count; ++page) {
                    ASSERT_EQ(pages_h.data_ptr<int32_t>()[offset + page],
                              block[static_cast<int64_t>(b) * columns + page]);
                }
                offset += count;
            } else {
                ASSERT_EQ(batches_h.data_ptr<int32_t>()[b], 0);
                ASSERT_EQ(kvlen_h.data_ptr<int32_t>()[b], 0);
                ASSERT_EQ(positions_h.data_ptr<int32_t>()[b], 0);
                ASSERT_EQ(last_h.data_ptr<int32_t>()[b], 0);
                ASSERT_EQ(slots_h.data_ptr<int64_t>()[b], -1);
                ASSERT_EQ(qo_h.data_ptr<int32_t>()[b + 1], batch);
            }
            ASSERT_EQ(indptr_h.data_ptr<int32_t>()[b + 1], offset);
        }
    }
};

class CudaGraphDecodeParamsTest: public ::testing::Test {
protected:
    cudaStream_t stream = nullptr;
    void         SetUp() override {
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    }
    void TearDown() override {
        cudaStreamDestroy(stream);
    }
};

TEST_F(CudaGraphDecodeParamsTest, ArbitraryBatchClampingPaddingAndInt64Slots) {
    for (int batch : {0, 1, 255, 256, 257, 768, 1024, 1025, 1152, 4097, 16385}) {
        for (int page_size : {0, 128}) {
            SCOPED_TRACE(::testing::Message() << "batch=" << batch << " page_size=" << page_size);
            DecodeBuffers t(batch + 7, 19);
            t.lengths.copy_(torch::arange(batch + 7, t.lengths.options()).remainder(3000) - 2);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            t.invoke(batch, page_size, stream);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            t.check(batch, page_size);
        }
    }
}

TEST_F(CudaGraphDecodeParamsTest, ReplayChangedLengthsAndShrinkAcrossDispatchBoundary) {
    DecodeBuffers t(1200, 1026);
    t.lengths.fill_(131073);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    t.invoke(1200, 128, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaGraph_t     graph = nullptr;
    cudaGraphExec_t exec  = nullptr;
    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
    t.invoke(1057, 128, stream);
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_EQ(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), cudaSuccess);
    for (int length : {131073, 129, 1, 65537, std::numeric_limits<int32_t>::max()}) {
        t.lengths.fill_(length);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(exec, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        t.check(1057, 128);
    }
    for (int batch : {768, 257, 256, 1, 0}) {
        t.invoke(batch, 128, stream);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        t.check(batch, 128);
    }
    ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

TEST_F(CudaGraphDecodeParamsTest, RejectInvalidLiveBatch) {
    DecodeBuffers t(8, 2);
    ASSERT_THROW(t.invoke(-1, 128, stream), c10::Error);
    ASSERT_THROW(t.invoke(9, 128, stream), c10::Error);
}

}  // namespace
}  // namespace rtp_llm
