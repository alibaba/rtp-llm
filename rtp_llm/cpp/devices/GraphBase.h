#pragma once
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

using namespace torch_ext;
class GraphBase {
public:
    GraphBase(py::object py_instance): py_instance_(std::move(py_instance)) {}
    virtual ~GraphBase() {}
    virtual void           initCapture()                                             = 0;
    virtual PyModelOutputs forward(PyModelInputs& inputs)                            = 0;
    virtual void           setPositionEncoding(torch::Tensor position_encoding)      = 0;
    virtual void           setTokenTypeEmbedding(torch::Tensor token_type_embedding) = 0;
    virtual void           setInputEmbeddingScalar(float input_embedding_scalar)     = 0;
    virtual void           setPositionIdLenFactor(int position_id_len_factor)        = 0;
    virtual void           setNeedComboPositionIds(bool need_combo_position_ids)     = 0;
    py::object             py_instance_;
};
}  // namespace rtp_llm
