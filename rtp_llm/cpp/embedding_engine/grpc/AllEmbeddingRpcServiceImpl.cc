#include "rtp_llm/cpp/embedding_engine/rpc/AllEmbeddingRpcServiceImpl.h"

namespace py = pybind11;
namespace th = torch;

namespace rtp_llm {
grpc::Status AllEmbeddingRpcServiceImpl::decode(grpc::ServerContext*     context,
                                                const AllEmbeddingInput* request,
                                                AllEmbeddingOutput*      writer) {
    py::gil_scoped_acquire acquire;
    py::list               lst;
    for (auto& request : request->input()) {
        lst.append(request);
    }
    py::list               render_result  = pyRenderer_.attr("render_cpp")(lst);
    th::Tensor             token_ids      = render_result[0].cast<th::Tensor>();
    th::Tensor             token_type_ids = render_result[1].cast<th::Tensor>();
    th::Tensor             input_lengths  = render_result[2].cast<th::Tensor>();
    int                    request_id     = 0;
    py::gil_scoped_release release;
    th::Tensor             out = embedding_engine_->decode(token_ids, token_type_ids, input_lengths, request_id);
    RTP_LLM_CHECK_WITH_INFO(out.dim() == 3, "all embedding out should be 3-dim");
    fill(writer, out);
    return grpc::Status::OK;
}

void AllEmbeddingRpcServiceImpl::fill(AllEmbeddingOutput* writer, torch::Tensor result) {
    writer->mutable_tensor()->mutable_shape()->Resize(result.dim(), 0);
    for (auto i = 0; i < result.dim(); i++) {
        (*writer->mutable_tensor()->mutable_shape())[i] = result.size(i);
    }
    writer->mutable_tensor()->set_data(reinterpret_cast<const char*>(result.mutable_data_ptr()), result.nbytes());
}
}  // namespace rtp_llm