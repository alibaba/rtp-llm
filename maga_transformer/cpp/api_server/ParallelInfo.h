#pragma once

#include "autil/EnvUtil.h"

namespace rtp_llm {

class ParallelInfo {
public:
    ParallelInfo(int world_size, int world_rank, int local_world_size):
        world_size_(world_size), world_rank_(world_rank), local_world_size_(local_world_size) {}

public:
    static ParallelInfo& globalParallelInfo() {
        int                 world_size       = autil::EnvUtil::getEnv("WORLD_SIZE", 1);
        int                 world_rank       = autil::EnvUtil::getEnv("WORLD_RANK", 0);
        int                 local_world_size = autil::EnvUtil::getEnv("LOCAL_WORLD_SIZE", 1);
        static ParallelInfo parallel_info(world_size, world_rank, local_world_size);
        return parallel_info;
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
    // only for test
    void reload() {
        world_size_       = autil::EnvUtil::getEnv("WORLD_SIZE", 1);
        world_rank_       = autil::EnvUtil::getEnv("WORLD_RANK", 0);
        local_world_size_ = autil::EnvUtil::getEnv("LOCAL_WORLD_SIZE", 1);
    }

private:
    int world_size_;
    int world_rank_;
    int local_world_size_;
};

}  // namespace rtp_llm