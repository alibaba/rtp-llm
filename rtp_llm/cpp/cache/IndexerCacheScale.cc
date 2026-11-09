#include "rtp_llm/cpp/cache/IndexerCacheScale.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace rtp_llm {
namespace {
bool known(IndexerScaleFormat format) {
    return format == IndexerScaleFormat::FP32 || format == IndexerScaleFormat::UE8M0;
}

struct ScaleViews {
    std::vector<at::Tensor> scales;
    std::vector<at::Tensor> payloads;
};

// Bound FP32 requantization scratch independently of the number of transferred pages.
constexpr int64_t kConversionRows = 4096;

at::Tensor fp32ScaleBits(const at::Tensor& exponent) {
    return at::where(exponent == 0, at::full_like(exponent, 0x00400000), at::bitwise_left_shift(exponent, 23));
}

void convertBatch(const std::vector<at::Tensor>& scale_views,
                  const std::vector<at::Tensor>& payload_views,
                  IndexerScaleFormat             source) {
    auto                 scales = at::cat(scale_views);
    at::Tensor           converted;
    std::vector<int64_t> sizes;
    for (const auto& view : scale_views)
        sizes.push_back(view.numel());
    if (source == IndexerScaleFormat::FP32) {
        auto raw_exponent = at::bitwise_and(at::bitwise_right_shift(scales, 23), 255);
        auto fraction     = at::bitwise_and(scales, 0x7fffff);
        // Round positive FP32 scales upward, including the UE8M0 subnormal boundary.
        auto exponent = at::where(raw_exponent == 0,
                                  (fraction > 0x400000).to(at::kInt),
                                  raw_exponent + (fraction != 0).to(at::kInt))
                            .clamp(0, 254);
        auto target_scale = fp32ScaleBits(exponent).view(at::kFloat);
        auto ratio        = scales.view(at::kFloat) / target_scale;
        // Avoid device FTZ changing subnormal FP32 ratios (including unused zero rows).
        auto subnormal_ratio =
            fraction.to(at::kFloat)
            * at::where(exponent == 0, at::full_like(target_scale, 0x1p-22f), at::full_like(target_scale, 0x1p-23f));
        ratio            = at::where(raw_exponent == 0, subnormal_ratio, ratio);
        auto payload     = at::cat(payload_views).view(at::kFloat8_e4m3fn).to(at::kFloat);
        auto requantized = (payload * ratio.unsqueeze(1)).clamp(-448, 448).to(at::kFloat8_e4m3fn).view(at::kByte);
        at::_foreach_copy_(payload_views, requantized.split_with_sizes(sizes), false);
        converted = exponent * 0x01010101;
    } else {
        converted = fp32ScaleBits(at::bitwise_and(scales, 255));
    }
    at::_foreach_copy_(scale_views, converted.split_with_sizes(sizes), false);
}
}  // namespace

IndexerCacheLayout indexerSourceLayout(const BlockInfo& destination, IndexerCacheLayout source) {
    if (source.format == IndexerScaleFormat::NONE && source.page_tokens == 0 && source.page_stride == 0
        && known(destination.indexer_layout.format)) {
        source        = destination.indexer_layout;
        source.format = IndexerScaleFormat::FP32;
    }
    return source;
}

IndexerTransferBuffer legacyIndexerTransfer(const BlockInfo& source, size_t alignment) {
    IndexerTransferBuffer result{source, source.size_bytes, {}};
    if (source.indexer_layout.format != IndexerScaleFormat::UE8M0)
        return result;
    validateIndexerScaleTransfer(source, source.indexer_layout);
    if (alignment == 0 || alignment > static_cast<size_t>(INT64_MAX)
        || source.size_bytes > static_cast<size_t>(INT64_MAX) - alignment + 1)
        throw std::runtime_error("invalid indexer transfer allocation alignment");
    const auto device = source.is_cuda ? at::globalContext().getDeviceFromPtr(source.addr, c10::DeviceType::CUDA) :
                                         c10::Device(c10::kCPU);
    c10::DeviceGuard guard(device);
    result.allocation_bytes = (source.size_bytes + alignment - 1) / alignment * alignment;
    auto owner              = std::make_shared<at::Tensor>(at::empty({static_cast<int64_t>(result.allocation_bytes)},
                                                        at::TensorOptions().dtype(at::kByte).device(device)));
    // Blocking copy; the existing cache is never modified or republished.
    owner->narrow(0, 0, source.size_bytes)
        .copy_(at::from_blob(source.addr, {static_cast<int64_t>(source.size_bytes)}, owner->options()));
    result.owner                       = std::shared_ptr<void>(owner, owner->data_ptr());
    result.block.addr                  = owner->data_ptr();
    result.block.indexer_layout.format = IndexerScaleFormat::FP32;
    convertIndexerCacheScales({{result.block, source.indexer_layout}});
    return result;
}

void validateIndexerScaleTransfer(const BlockInfo& destination, const IndexerCacheLayout& wire_source) {
    const auto  source = indexerSourceLayout(destination, wire_source);
    const auto& target = destination.indexer_layout;
    if (source.format == IndexerScaleFormat::NONE && target.format == IndexerScaleFormat::NONE)
        return;
    if (!known(source.format) || !known(target.format) || !destination.addr || source.page_tokens == 0
        || source.page_tokens != target.page_tokens || source.page_stride != target.page_stride
        || source.page_stride % 4 != 0 || source.page_stride < static_cast<uint64_t>(source.page_tokens) * 132
        || destination.size_bytes == 0 || destination.size_bytes % source.page_stride != 0
        || reinterpret_cast<uintptr_t>(destination.addr) % 4 != 0) {
        throw std::runtime_error("incompatible FP8 indexer scale encoding or kernel-page geometry");
    }
}

void convertIndexerCacheScales(const std::vector<IndexerScaleConversion>& blocks) {
    using Key = std::tuple<int, int, uint32_t>;
    std::map<Key, ScaleViews> groups;
    for (const auto& [destination, wire_source] : blocks) {
        const auto source = indexerSourceLayout(destination, wire_source);
        validateIndexerScaleTransfer(destination, source);
        if (source.format == destination.indexer_layout.format)
            continue;
        const auto device = destination.is_cuda ?
                                at::globalContext().getDeviceFromPtr(destination.addr, c10::DeviceType::CUDA) :
                                c10::Device(c10::kCPU);
        auto& views = groups[{static_cast<int>(device.type()), device.index(), static_cast<uint32_t>(source.format)}];
        auto* base  = static_cast<uint8_t*>(destination.addr);
        for (size_t offset = 0; offset < destination.size_bytes; offset += source.page_stride) {
            for (int64_t row = 0; row < source.page_tokens; row += kConversionRows) {
                const int64_t rows = std::min(kConversionRows, static_cast<int64_t>(source.page_tokens) - row);
                views.scales.push_back(
                    at::from_blob(base + offset + static_cast<size_t>(source.page_tokens) * 128 + row * 4,
                                  {rows},
                                  at::TensorOptions().dtype(at::kInt).device(device)));
                if (source.format == IndexerScaleFormat::FP32) {
                    views.payloads.push_back(at::from_blob(
                        base + offset + row * 128, {rows, 128}, at::TensorOptions().dtype(at::kByte).device(device)));
                }
            }
        }
    }
    for (const auto& [key, views] : groups) {
        const auto                  device = views.scales.front().device();
        c10::DeviceGuard            guard(device);
        c10::impl::VirtualGuardImpl impl(device.type());
        try {
            for (size_t begin = 0; begin < views.scales.size();) {
                size_t  end  = begin;
                int64_t rows = 0;
                while (end < views.scales.size() && rows + views.scales[end].numel() <= kConversionRows) {
                    rows += views.scales[end++].numel();
                }
                std::vector<at::Tensor> scales(views.scales.begin() + begin, views.scales.begin() + end);
                std::vector<at::Tensor> payloads;
                if (!views.payloads.empty())
                    payloads.assign(views.payloads.begin() + begin, views.payloads.begin() + end);
                convertBatch(scales, payloads, static_cast<IndexerScaleFormat>(std::get<2>(key)));
                begin = end;
            }
        } catch (...) {
            // Queued work still owns target bytes if a later allocation/launch fails.
            if (device.type() != c10::kCPU)
                impl.synchronizeStream(impl.getStream(device));
            throw;
        }
        if (device.type() != c10::kCPU)
            impl.synchronizeStream(impl.getStream(device));
    }
}

}  // namespace rtp_llm
