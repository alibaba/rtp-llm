#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

class PrefillRequestLogger {
public:
    PrefillRequestLogger(const std::string& log_path,
                         int64_t            rank_id,
                         int                backup_count,
                         size_t             max_file_bytes = 256ULL * 1024 * 1024);
    ~PrefillRequestLogger();
    void log(const GenerateInputPB& request) const;

private:
    class Worker;
    std::unique_ptr<Worker> worker_;
    int64_t                 rank_id_;
};

}  // namespace rtp_llm
