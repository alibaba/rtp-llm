#pragma once

#include "autil/EnvUtil.h"

namespace rtp_llm {

class ParallelInfo final {
public:
    ParallelInfo(int tp_size,
                 int pp_size,
                 int ep_size,
                 int dp_size,
                 int world_size,
                 int world_rank,
                 int local_world_size):
        tp_size_(tp_size),
        pp_size_(pp_size),
        ep_size_(ep_size),
        dp_size_(dp_size),
        world_size_(world_size),
        world_rank_(world_rank),
        local_world_size_(local_world_size) {}

public:
    static ParallelInfo& globalParallelInfo() {

        int tp_size          = autil::EnvUtil::getEnv("TP_SIZE", 1);
        int pp_size          = autil::EnvUtil::getEnv("PP_SIZE", 1);
        int ep_size          = autil::EnvUtil::getEnv("EP_SIZE", 1);
        int dp_size          = autil::EnvUtil::getEnv("DP_SIZE", 1);
        int world_size       = autil::EnvUtil::getEnv("WORLD_SIZE", 1);
        int world_rank       = autil::EnvUtil::getEnv("WORLD_RANK", 0);
        int local_world_size = autil::EnvUtil::getEnv("LOCAL_WORLD_SIZE", 1);

        static ParallelInfo parallel_info(tp_size, pp_size, ep_size, dp_size, world_size, world_rank, local_world_size);
        return parallel_info;
    }
    int getTpSize() const {
        return tp_size_;
    }
    int getPpSize() const {
        return pp_size_;
    }
    int getDpSize() const {
        return dp_size_;
    }
    int getEpSize() const {
        return ep_size_;
    }
    int getTpRank() const {
        return world_rank_ % tp_size_;
    }
    int getWorldRank() const {
        return world_rank_;
    }
    int getLocalRank() const {
        return world_rank_ % local_world_size_;
    }
    int getWorldSize() const {
        return world_size_;
    }

    int getLocalWorldSize() const {
        return local_world_size_;
    }
    bool isMaster() const {
        return world_rank_ == 0;
    }
    bool isWorker() const {
        return !isMaster();
    }
    std::string toString() const {
        std::ostringstream oss;
        oss << "ParallelInfo:[ "
            << "tp_size=" << tp_size_ << " pp_size=" << pp_size_ << " world_size=" << world_size_
            << " world_rank=" << world_rank_ << " local_world_size=" << local_world_size_ << " ]";
        return oss.str();
    }
    // only for test
    void reload() {
        tp_size_          = autil::EnvUtil::getEnv("TP_SIZE", 1);
        pp_size_          = autil::EnvUtil::getEnv("PP_SIZE", 1);
        dp_size_          = autil::EnvUtil::getEnv("DP_SIZE", 1);
        ep_size_          = autil::EnvUtil::getEnv("EP_SIZE", 1);
        world_size_       = autil::EnvUtil::getEnv("WORLD_SIZE", 1);
        world_rank_       = autil::EnvUtil::getEnv("WORLD_RANK", 0);
        local_world_size_ = autil::EnvUtil::getEnv("LOCAL_WORLD_SIZE", 1);
    }

private:
    int         tp_size_;
    int         pp_size_;
    int         ep_size_;
    int         dp_size_;
    int         world_size_;
    int         world_rank_;
    int         local_world_size_;
};

}  // namespace rtp_llm
