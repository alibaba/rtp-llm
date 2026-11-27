#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rtp_llm {

using CacheKeyType = int64_t;
using CacheKeysType = std::vector<CacheKeyType>;

class BlockTreeKeyView {
public:
    BlockTreeKeyView(CacheKeyType key): integer_(key) {}
    explicit BlockTreeKeyView(const std::string& bytes): bytes_(&bytes) {}

    bool isOpaque() const { return bytes_ != nullptr; }
    CacheKeyType integer() const {
        assert(!isOpaque() && "An opaque cache key has no integer representation");
        return integer_;
    }
    std::string_view bytes() const {
        return bytes_ ? std::string_view(*bytes_) : std::string_view{};
    }
    bool operator==(const BlockTreeKeyView& other) const {
        return isOpaque() == other.isOpaque()
            && (isOpaque() ? bytes() == other.bytes() : integer_ == other.integer_);
    }

private:
    CacheKeyType integer_{0};
    const std::string* bytes_{nullptr};
};

struct BlockTreeKeyHash {
    size_t operator()(BlockTreeKeyView key) const {
        return key.isOpaque() ? std::hash<std::string_view>{}(key.bytes()) : std::hash<CacheKeyType>{}(key.integer());
    }
};

class BlockTreeKey {
public:
    explicit BlockTreeKey(CacheKeyType key): integer_(key) {}
    explicit BlockTreeKey(std::string_view bytes): bytes_(bytes) {
        if (bytes.empty()) {
            throw std::invalid_argument("Opaque cache keys must be nonempty");
        }
    }
    explicit BlockTreeKey(BlockTreeKeyView key):
        integer_(key.isOpaque() ? 0 : key.integer()), bytes_(key.bytes()) {}

    operator BlockTreeKeyView() const {
        return bytes_.empty() ? BlockTreeKeyView(integer_) : BlockTreeKeyView(bytes_);
    }
    bool isOpaque() const { return !bytes_.empty(); }
    CacheKeyType integer() const { return static_cast<BlockTreeKeyView>(*this).integer(); }
    std::string_view bytes() const { return bytes_; }

private:
    CacheKeyType integer_{0};
    std::string bytes_;
};

using BlockTreeKeys = std::vector<BlockTreeKey>;

}  // namespace rtp_llm
