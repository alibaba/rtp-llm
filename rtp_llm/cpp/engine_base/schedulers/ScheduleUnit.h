#pragma once

#include <cstdint>
#include <vector>
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

struct ScheduleUnit {
    int64_t                        group_id = -1;
    std::vector<GenerateStreamPtr> streams;

    bool prepare() {
        bool any_loading = false;
        for (auto it = streams.begin(); it != streams.end();) {
            bool needs = (*it)->prepare();
            if (!(*it)->alive()) {
                it = streams.erase(it);
                continue;
            }
            if (needs)
                any_loading = true;
            ++it;
        }
        return any_loading;
    }

    bool isReady() {
        for (auto& s : streams) {
            if (!s->isReady())
                return false;
        }
        return true;
    }

    void activate() {
        for (auto it = streams.begin(); it != streams.end();) {
            (*it)->activate();
            if (!(*it)->alive()) {
                it = streams.erase(it);
                continue;
            }
            ++it;
        }
    }

    void advance() {
        for (auto it = streams.begin(); it != streams.end();) {
            (*it)->advance();
            if (!(*it)->alive()) {
                it = streams.erase(it);
                continue;
            }
            ++it;
        }
    }

    bool alive() const {
        return !streams.empty();
    }

    bool hasError() const {
        for (const auto& s : streams) {
            if (s->hasError()) {
                return true;
            }
        }
        return false;
    }

    bool isGroup() const {
        return group_id != -1;
    }

    size_t size() const {
        return streams.size();
    }
};

}  // namespace rtp_llm
