#include "maga_transformer/cpp/disaggregate/rtpllm_master/entry/RtpLLMMasterEntry.h"
#include "maga_transformer/cpp/disaggregate/rtpllm_master/entry/MasterInitParameter.h"

namespace rtp_llm {
namespace rtp_llm_master {

PYBIND11_MODULE(librtpllm_master, m) {
    rtp_llm::rtp_llm_master::registerRtpLLMMasterEntry(m);
    rtp_llm::rtp_llm_master::registerMasterInitParameter(m);
}

}  // namespace rtp_llm_master
}  // namespace rtp_llm