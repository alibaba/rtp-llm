#pragma once

#include <functional>
#include <string>
#include <tuple>

#include "rtp_llm/cpp/cache/CacheModuleId.h"

namespace rtp_llm {

// A tag is unique only inside its owning module. Never encode the module in
// the tag: model bindings and transport must retain the original local tag.
struct ModuleGroupKey {
    CacheModuleId module_id;
    std::string   tag;

    std::string debugString() const {
        return "module=" + std::to_string(module_id.value()) + " tag=" + tag;
    }

    friend bool operator==(const ModuleGroupKey& lhs, const ModuleGroupKey& rhs) {
        return lhs.module_id == rhs.module_id && lhs.tag == rhs.tag;
    }

    friend bool operator!=(const ModuleGroupKey& lhs, const ModuleGroupKey& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator<(const ModuleGroupKey& lhs, const ModuleGroupKey& rhs) {
        return std::tie(lhs.module_id, lhs.tag) < std::tie(rhs.module_id, rhs.tag);
    }
};

}  // namespace rtp_llm

namespace std {
template<>
struct hash<rtp_llm::ModuleGroupKey> {
    size_t operator()(const rtp_llm::ModuleGroupKey& key) const noexcept {
        const size_t module_hash = hash<rtp_llm::CacheModuleId>()(key.module_id);
        const size_t tag_hash    = hash<string>()(key.tag);
        return module_hash ^ (tag_hash + size_t{0x9e3779b9} + (module_hash << 6) + (module_hash >> 2));
    }
};
}  // namespace std
