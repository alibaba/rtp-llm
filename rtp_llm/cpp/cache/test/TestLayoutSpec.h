#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm::test {

// Explicit layouts for tests of buffer strides and transfer boundaries.
struct TestLayoutSpec: KVCacheSpec {
    TestLayoutSpec(const KVCacheSpec& source, size_t kv_stride, size_t scale_stride):
        KVCacheSpec(source),
        elems(source.block_size()),
        k_elems(source.k_block_size()),
        v_elems(source.v_block_size()),
        kv_bytes(kv_stride),
        k_bytes(source.k_block_size_bytes()),
        v_bytes(source.v_block_size_bytes()),
        payload(source.block_payload_bytes()),
        k_payload(source.k_block_payload_bytes()),
        v_payload(source.v_block_payload_bytes()),
        scale_bytes(scale_stride),
        k_scale_bytes(source.k_scale_block_size_bytes()),
        v_scale_bytes(source.v_scale_block_size_bytes()),
        dtype(source.memoryLayoutDType()) {}

    size_t block_size() const override {
        return elems;
    }
    size_t k_block_size() const override {
        return k_elems;
    }
    size_t v_block_size() const override {
        return v_elems;
    }
    size_t block_size_bytes() const override {
        return kv_bytes;
    }
    size_t k_block_size_bytes() const override {
        return k_bytes;
    }
    size_t v_block_size_bytes() const override {
        return v_bytes;
    }
    size_t block_payload_bytes() const override {
        return payload;
    }
    size_t k_block_payload_bytes() const override {
        return k_payload;
    }
    size_t v_block_payload_bytes() const override {
        return v_payload;
    }
    size_t scale_block_size_bytes() const override {
        return scale_bytes;
    }
    size_t k_scale_block_size_bytes() const override {
        return k_scale_bytes;
    }
    size_t v_scale_block_size_bytes() const override {
        return v_scale_bytes;
    }
    DataType memoryLayoutDType() const override {
        return dtype;
    }
    KVCacheSpecPtr clone() const override {
        return std::make_shared<TestLayoutSpec>(*this);
    }
    std::string debugString(size_t indent = 0) const override {
        return commonDebugString(indent);
    }

    size_t   elems, k_elems, v_elems;
    size_t   kv_bytes, k_bytes, v_bytes;
    size_t   payload, k_payload, v_payload;
    size_t   scale_bytes, k_scale_bytes, v_scale_bytes;
    DataType dtype;
};

inline void setGroupLayout(GroupBase& group, size_t kv_stride, size_t scale_stride) {
    RTP_LLM_CHECK(group.spec != nullptr);
    if (group.spec->block_size_bytes() == kv_stride && group.spec->scale_block_size_bytes() == scale_stride) {
        return;
    }
    group.spec = std::make_shared<TestLayoutSpec>(*group.spec, kv_stride, scale_stride);
}

inline void setGroupBlockLayout(CacheConfig&                 config,
                                const std::vector<uint32_t>& block_nums,
                                const std::vector<size_t>&   kv_strides,
                                const std::vector<size_t>&   scale_strides) {
    auto groups = config.topology().groups();
    RTP_LLM_CHECK(groups.size() == block_nums.size());
    RTP_LLM_CHECK(groups.size() == kv_strides.size());
    RTP_LLM_CHECK(groups.size() == scale_strides.size());
    for (size_t gid = 0; gid < groups.size(); ++gid) {
        groups[gid].block_num = block_nums[gid];
        setGroupLayout(groups[gid], kv_strides[gid], scale_strides[gid]);
    }
    config.setTopology(std::move(groups), config.topology().layers());
    config.setGroupBlockLayout(block_nums, kv_strides, scale_strides);
}

}  // namespace rtp_llm::test
