#pragma once

#include "src/fastertransformer/devices/Weights.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"

namespace ft = fastertransformer;

namespace rtp_llm {

std::unique_ptr<const ft::Weights> convertPythonWeights(const PyModelWeights& weights);

}
