#pragma once
#include "maga_transformer/cpp/disaggregate/master/scheduler/Struct.h"
#include "maga_transformer/cpp/disaggregate/master/cluster/PrefillCluster.h"
#include "maga_transformer/cpp/disaggregate/master/scheduler/PrefillScheduler.h"
#include "absl/status/statusor.h"

namespace rtp_llm {
namespace rtp_llm_master {

class Entry {
public:
    //TODO: 是否需要解析请求?
    std::string handleRequest(const std::string& request);
protected:
    //TODO: 怎么去设计接口，能够让tokenize只进行一遍，但是能获得所有信息？
    RequestInfo tokenizeAndHash(const std::string& request);
private:
    std::unique_ptr<PrefillScheduler> scheduler_;
};

}  // namespace rtp_llm_master
}  // namespace rtp_llm