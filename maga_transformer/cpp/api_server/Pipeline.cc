#include "maga_transformer/cpp/api_server/Pipeline.h"

namespace rtp_llm {

Pipeline::Pipeline(py::object token_processor): token_processor_(token_processor) {}

}  // namespace rtp_llm
