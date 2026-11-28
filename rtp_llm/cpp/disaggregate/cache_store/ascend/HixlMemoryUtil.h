// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlProtocol.h"

#include <acl/acl.h>
#include <hixl/hixl.h>
#include <map>
#include <mutex>

namespace rtp_llm {

// The transport owns the initialized engine and its ACL context. It freezes
// this registry before connecting and thaws it only after all peers disconnect.
class HixlMemoryUtil final: public MemoryUtil {
public:
    HixlMemoryUtil(std::shared_ptr<hixl::Hixl> engine, aclrtContext context);
    ~HixlMemoryUtil() override;
    static std::shared_ptr<HixlMemoryUtil> create(const std::string& engine_id, const std::string& local_comm_res);
    const std::shared_ptr<hixl::Hixl>&     engine() const {
        return engine_;
    }
    aclrtContext context() const {
        return context_;
    }
    bool regUserMr(void* buf, uint64_t size, bool gpu, uint64_t aligned_size = 0) override;
    bool regUserMrWithOwner(void*                        buf,
                            uint64_t                     size,
                            bool                         gpu,
                            uint64_t                     aligned_size,
                            const std::shared_ptr<void>& owner) override;
    bool deregUserMr(void* buf, bool gpu) override;
    bool isMemoryMr(void* ptr, uint64_t size, bool gpu, bool adopted) override;
    bool findMemoryMr(void* info, void* ptr, uint64_t size, bool gpu, bool adopted) override;
    bool isRdmaMode() override {
        return false;
    }
    bool supportsDeviceMemory() override {
        return true;
    }
    bool requiresBlockLease() const override {
        return true;
    }

    void freezeRegistrations();
    void thawAfterDisconnect();

private:
    struct Registration {
        HixlMemoryRegion region;
        hixl::MemHandle  handle{nullptr};
        // Keep the allocation alive until SDK deregistration and shutdown complete.
        std::shared_ptr<void> owner;
    };

    const Registration* find(uintptr_t address, uint64_t size) const;

    std::shared_ptr<hixl::Hixl>       engine_;
    aclrtContext                      context_;
    std::mutex                        mutex_;
    std::map<uintptr_t, Registration> registrations_;
    uint64_t                          next_id_{1};
    bool                              frozen_{false};
};

}  // namespace rtp_llm
