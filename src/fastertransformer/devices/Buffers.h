#pragma once

#include "src/fastertransformer/core/Tensor.h"

namespace fastertransformer {

// OpBuffer is an abstract class for all buffers used in operations
// Each device need to implement its own buffer class for each operation
class OpBuffer {
public:
    virtual ~OpBuffer() {};
};

}  // namespace fastertransformer
