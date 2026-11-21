// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlMemoryUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace rtp_llm {
namespace {

bool validRange(uintptr_t address, uint64_t size) {
    return address != 0 && size != 0 && size <= std::numeric_limits<uintptr_t>::max() - address;
}

bool checked(uint32_t status, const char* operation) {
    if (status != 0) {
        RTP_LLM_LOG_ERROR("HIXL %s failed, status=%u", operation, status);
    }
    return status == 0;
}

}  // namespace

HixlMemoryUtil::HixlMemoryUtil(std::shared_ptr<hixl::Hixl> engine, aclrtContext context):
    engine_(std::move(engine)), context_(context) {}

std::shared_ptr<HixlMemoryUtil> HixlMemoryUtil::create(const std::string& engine_id,
                                                       const std::string& local_comm_res) {
    if (local_comm_res.empty()) {
        throw std::invalid_argument("HIXL requires LocalCommRes configuration");
    }
    aclrtContext context = nullptr;
    if (!checked(aclrtGetCurrentContext(&context), "get initialization context") || !context) {
        throw std::runtime_error("HIXL requires an initialized device context");
    }
    auto engine = std::make_shared<hixl::Hixl>();
    // Eight channels reserve 16 streams, independently of the 64 admitted reads.
    if (!checked(engine->Initialize(hixl::AscendString(engine_id.c_str()),
                                    {{hixl::OPTION_LOCAL_COMM_RES, hixl::AscendString(local_comm_res.c_str())},
                                     {hixl::OPTION_GLOBAL_RESOURCE_CONFIG,
                                      hixl::AscendString(R"({"comm_resource_config.max_active_channels":"8"})")}}),
                 "Initialize")) {
        throw std::runtime_error("HIXL initialization failed");
    }
    return std::make_shared<HixlMemoryUtil>(std::move(engine), context);
}

HixlMemoryUtil::~HixlMemoryUtil() {
    if (!checked(aclrtSetCurrentContext(context_), "restore finalize context")) {
        std::terminate();
    }
    for (const auto& item : registrations_) {
        if (!checked(engine_->DeregisterMem(item.second.handle), "deregister finalized pool")) {
            std::terminate();
        }
    }
    engine_->Finalize();
}

bool HixlMemoryUtil::regUserMr(void* buf, uint64_t size, bool gpu, uint64_t aligned_size) {
    return regUserMrWithOwner(buf, size, gpu, aligned_size, {});
}

bool HixlMemoryUtil::regUserMrWithOwner(
    void* buf, uint64_t size, bool gpu, uint64_t, const std::shared_ptr<void>& owner) {
    const auto                  address = reinterpret_cast<uintptr_t>(buf);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!gpu || !validRange(address, size)) {
        return false;
    }
    auto next = registrations_.lower_bound(address);
    if (next != registrations_.end() && next->first == address && next->second.region.bytes == size) {
        if (owner) {
            next->second.owner = owner;
        }
        return true;
    }
    if (frozen_ || (next != registrations_.end() && next->first < address + size)) {
        return false;
    }
    if (next != registrations_.begin()) {
        const auto& previous = std::prev(next)->second.region;
        if (previous.address + previous.bytes > address) {
            return false;
        }
    }
    if (!checked(aclrtSetCurrentContext(context_), "restore context")) {
        return false;
    }
    hixl::MemHandle handle = nullptr;
    if (!checked(engine_->RegisterMem({address, size}, hixl::MEM_DEVICE, handle), "RegisterMem")) {
        return false;
    }
    registrations_.emplace(address, Registration{{address, size, next_id_++}, handle, owner});
    return true;
}

bool HixlMemoryUtil::deregUserMr(void* buf, bool gpu) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = registrations_.find(reinterpret_cast<uintptr_t>(buf));
    if (!gpu || frozen_ || it == registrations_.end()) {
        return false;
    }
    if (!checked(aclrtSetCurrentContext(context_), "restore context")
        || !checked(engine_->DeregisterMem(it->second.handle), "DeregisterMem")) {
        return false;
    }
    registrations_.erase(it);
    return true;
}

const HixlMemoryUtil::Registration* HixlMemoryUtil::find(uintptr_t address, uint64_t size) const {
    if (!validRange(address, size)) {
        return nullptr;
    }
    auto it = registrations_.upper_bound(address);
    if (it == registrations_.begin()) {
        return nullptr;
    }
    const auto& registration = std::prev(it)->second;
    const auto  offset       = address - registration.region.address;
    return offset <= registration.region.bytes && size <= registration.region.bytes - offset ? &registration : nullptr;
}

bool HixlMemoryUtil::isMemoryMr(void* ptr, uint64_t size, bool gpu, bool) {
    std::lock_guard<std::mutex> lock(mutex_);
    return gpu && find(reinterpret_cast<uintptr_t>(ptr), size) != nullptr;
}

bool HixlMemoryUtil::findMemoryMr(void* info, void* ptr, uint64_t size, bool gpu, bool) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto*                 registration = gpu ? find(reinterpret_cast<uintptr_t>(ptr), size) : nullptr;
    if (!registration) {
        return false;
    }
    *static_cast<HixlMemoryRegion*>(info) = registration->region;
    return true;
}

void HixlMemoryUtil::freezeRegistrations() {
    std::lock_guard<std::mutex> lock(mutex_);
    frozen_ = true;
}

void HixlMemoryUtil::thawAfterDisconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    frozen_ = false;
}

}  // namespace rtp_llm
