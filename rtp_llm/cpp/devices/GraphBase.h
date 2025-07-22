#pragma once
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

using namespace torch_ext;
class GraphBase {
public:
    GraphBase(py::object py_instance): py_instance_(py_instance) {}
    virtual void           capture()                            = 0;
    virtual void           captureOneBatchSize(int bs)          = 0;
    virtual void           prepareInputs(PyModelInputs& inputs) = 0;
    virtual bool           canRun(PyModelInputs& inputs)        = 0;
    virtual void           replay(int bs)                       = 0;
    virtual void           initCapture()                        = 0;
    virtual PyModelOutputs forward(PyModelInputs& inputs)       = 0;
    py::object             py_instance_;
};
}  // namespace rtp_llm
