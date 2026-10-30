#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"

#include "rtp_llm/cpp/cache/CPSlotMapper.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

namespace rtp_llm {

namespace {

// Planning expands remote rank metadata synchronously on the scheduler path.
constexpr int64_t kMaxPlanRankCount = 1024;

ErrorInfo planError(const std::string& msg) {
    return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, "transfer plan: " + msg);
}

std::string sideName(bool is_src) {
    return is_src ? "src" : "dst";
}

bool supportsAsymmetricPageRR(const ShardLayout& src, const ShardLayout& dst, const std::string& tag) {
    const auto src_cp = src.routeCpSize(tag);
    const auto dst_cp = dst.routeCpSize(tag);
    return src.local_full_page_rr && dst.local_full_page_rr && src_cp > 0 && dst_cp > 0
           && src.effectiveMapping(tag) == CpBlockMappingMode::BLOCK_ROUND_ROBIN
           && dst.effectiveMapping(tag) == CpBlockMappingMode::BLOCK_ROUND_ROBIN
           && src.group(tag).policy.active_tail_blocks == 0 && std::max(src_cp, dst_cp) % std::min(src_cp, dst_cp) == 0;
}

bool validLinearQkvPayload(const ShardLayout::GroupLayout& group) {
    if (group.kv_scale_stride_bytes != 0 || group.pre_sliced || group.linear_history_rows == 0
        || group.k_block_payload_bytes == 0 || group.k_block_payload_bytes >= group.kv_block_stride_bytes) {
        return false;
    }
    const auto conv_bytes = group.kv_block_stride_bytes - group.k_block_payload_bytes;
    const auto row_bytes  = conv_bytes / group.linear_history_rows;
    return conv_bytes % group.linear_history_rows == 0 && group.linear_q_row_bytes > 0 && group.linear_k_row_bytes > 0
           && group.linear_v_row_bytes > 0 && group.linear_q_row_bytes <= row_bytes
           && group.linear_k_row_bytes <= row_bytes - group.linear_q_row_bytes
           && group.linear_v_row_bytes == row_bytes - group.linear_q_row_bytes - group.linear_k_row_bytes;
}

bool compatibleLinearQkvLayouts(const ShardLayout& src, const ShardLayout& dst, const std::string& tag) {
    const auto& sg = src.group(tag);
    const auto& dg = dst.group(tag);
    const auto  sh = src.headShardCount(tag);
    const auto  dh = dst.headShardCount(tag);
    if (sh <= 0 || dh <= 0 || !validLinearQkvPayload(sg) || !validLinearQkvPayload(dg)
        || sg.linear_history_rows != dg.linear_history_rows || std::max(sh, dh) % std::min(sh, dh) != 0) {
        return false;
    }

    // The side with fewer head shards owns a larger local block. Verify each
    // transferred component, not just the total byte count.
    const auto& large   = sh < dh ? sg : dg;
    const auto& small   = sh < dh ? dg : sg;
    const auto  ratio   = static_cast<size_t>(std::max(sh, dh) / std::min(sh, dh));
    const auto  matches = [ratio](size_t wide, size_t narrow) { return wide % ratio == 0 && wide / ratio == narrow; };
    return matches(large.k_block_payload_bytes, small.k_block_payload_bytes)
           && matches(large.linear_q_row_bytes, small.linear_q_row_bytes)
           && matches(large.linear_k_row_bytes, small.linear_k_row_bytes)
           && matches(large.linear_v_row_bytes, small.linear_v_row_bytes);
}

// Sub-block count produced by MemoryLayoutStrategy for one block:
//  - head 不分片的 spec（MLA / Opaque）走 createBasicBlockInfo -> 1 个子块（+scale）
//  - head 分片的 spec 走 createPartitionedBlockInfo -> K/V 两个子块（+scale 两个）
int subBlockCount(const ShardLayout& side, const std::string& tag) {
    const auto& g     = side.group(tag);
    const int   base  = ShardLayout::specShardsHeads(g.spec_type) ? 2 : 1;
    const int   scale = g.kv_scale_stride_bytes > 0 ? base : 0;
    return base + scale;
}

// 一侧在一条 route 上实际参与传输的字节数。
//
// 注意两种 slice 模式的**分母不同**，必须与 CPSlotMapper::sliceBlockForPeer 一致：
//   EQUAL_BYTES   -> block.size_bytes / cp_size （整个 block stride）
//   PAYLOAD_BYTES -> spec->k_block_payload_bytes() / cp_size （只覆盖 payload 区间，
//                    stride 的对齐填充尾部不参与传输）
// 一律除 stride 会在 stride > payload（blockStrideBytes 做了对齐上取整）时算错。
size_t viewBytes(const ShardLayout::GroupLayout& group, const PartitionSpec& partition, const SliceSpec& slice) {
    const size_t partition_count = static_cast<size_t>(std::max(1, partition.count));
    const size_t slice_count     = static_cast<size_t>(std::max(1, slice.count));

    if (slice.mode == CpBlockSliceMode::NONE || slice_count == 1) {
        return group.kv_block_stride_bytes / partition_count;
    }
    if (slice.mode == CpBlockSliceMode::PAYLOAD_BYTES) {
        const size_t payload = group.k_block_payload_bytes;
        if (payload % partition_count != 0 || (payload / partition_count) % slice_count != 0) {
            return 0;
        }
        return payload / partition_count / slice_count;
    }
    return group.kv_block_stride_bytes / partition_count / slice_count;
}

}  // namespace

std::vector<size_t> KVCacheTransferPlanner::resolveKeys(const KeyShardSpec& spec, size_t logical_count) {
    std::vector<size_t> out;
    if (logical_count == 0) {
        return out;
    }
    const size_t modulus = static_cast<size_t>(std::max(1, spec.modulus));
    const size_t residue  = static_cast<size_t>(std::max(0, spec.residue)) % modulus;

    const size_t split_count = static_cast<size_t>(std::max(1, spec.replica_split_count));
    const size_t split_index = static_cast<size_t>(std::max(0, spec.replica_split_index));

    for (size_t pos = residue; pos < logical_count; pos += modulus) {
        // 副本均分（Step 3b）：默认 split_count == 1，该判断恒为真。
        if (split_count > 1 && (pos / modulus) % split_count != split_index) {
            continue;
        }
        out.push_back(pos);
    }

    // COMPACT 的非周期部分：序列末位无论落在哪个剩余类都必须被取到。
    if (spec.include_final_key) {
        const size_t final_pos = logical_count - 1;
        if (out.empty() || out.back() != final_pos) {
            out.push_back(final_pos);
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
        }
    }

    // active_tail_blocks：只保留末尾 tail_count 个。**在这里做**（而不是两侧执行期各自筛），
    // 因为 buildCacheStorePlan 的 start = total - tail_count 里的 total 是本侧块数，两侧
    // compact 程度不同时"最后 N 个"会指向不同的 key。out 已是本 route 的键集且升序，
    // 对其取尾在两侧一致。
    if (spec.tail_count > 0 && out.size() > static_cast<size_t>(spec.tail_count)) {
        out.erase(out.begin(), out.end() - spec.tail_count);
    }
    return out;
}

bool KVCacheTransferPlanner::ownsPosition(
    const ShardLayout& side, const std::string& tag, size_t pos, size_t count, int cp_rank) {
    const int cp_size = side.routeCpSize(tag);
    if (cp_size <= 0 || cp_rank < 0 || cp_rank >= cp_size) {
        return false;
    }
    if (side.group(tag).policy.group_type != CacheGroupType::FULL) {
        const size_t span = side.group(tag).seq_size_per_block / side.cache_key_tokens_per_block;
        return pos % span == span - 1 || pos + 1 == count;
    }
    CacheGroupPolicy policy = side.group(tag).policy;
    // 镜像 CPSlotMapper::layoutForGroup：cpSize() <= 1 时 mapping 退化为 NONE。
    policy.cp_mapping = side.effectiveMapping(tag);
    return CPSlotMapper::physicalBlockPosition(policy, pos, count, cp_rank, side.physicalCpSize(tag)).has_value();
}

ErrorInfo
KVCacheTransferPlanner::validateTag(const ShardLayout& src, const ShardLayout& dst, const std::string& tag) {
    if (!src.hasGroup(tag) || !dst.hasGroup(tag)) {
        return planError("tag '" + tag + "' missing on " + (src.hasGroup(tag) ? "dst" : "src") + " layout");
    }
    const auto& sg = src.group(tag);
    const auto& dg = dst.group(tag);

    if (sg.policy.group_type != dg.policy.group_type) {
        return planError(std::string("group_type mismatch for tag '") + tag + "': src="
                         + cacheGroupTypeName(sg.policy.group_type) + " dst=" + cacheGroupTypeName(dg.policy.group_type));
    }
    if (src.cache_key_tokens_per_block == 0 || src.cache_key_tokens_per_block != dst.cache_key_tokens_per_block) {
        return planError("cache-key token stride mismatch or zero for tag '" + tag
                         + "': src=" + std::to_string(src.cache_key_tokens_per_block)
                         + " dst=" + std::to_string(dst.cache_key_tokens_per_block));
    }
    if (sg.seq_size_per_block == 0 || sg.seq_size_per_block != dg.seq_size_per_block
        || sg.seq_size_per_block % src.cache_key_tokens_per_block != 0) {
        return planError("group token stride mismatch or invalid key coverage for tag '" + tag + "': src="
                         + std::to_string(sg.seq_size_per_block) + " dst=" + std::to_string(dg.seq_size_per_block)
                         + " key_tokens=" + std::to_string(src.cache_key_tokens_per_block));
    }
    const size_t key_span = sg.seq_size_per_block / src.cache_key_tokens_per_block;
    if (key_span > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return planError("group key span exceeds supported range for tag '" + tag + "'");
    }
    if (sg.policy.group_type == CacheGroupType::FULL && key_span != 1) {
        return planError("FULL P2P group must cover exactly one cache key per block for tag '" + tag + "'");
    }

    if (sg.policy.active_tail_blocks != dg.policy.active_tail_blocks) {
        std::ostringstream oss;
        oss << "active_tail_blocks mismatch for tag '" << tag << "': src=" << sg.policy.active_tail_blocks
            << " dst=" << dg.policy.active_tail_blocks
            << "; the tail restriction is folded into KeyShardSpec and must be identical on both sides";
        return planError(oss.str());
    }
    if (sg.spec_type != dg.spec_type) {
        return planError(std::string("spec_type mismatch for tag '") + tag
                         + "': src=" + KVCacheSpecTypeToString(sg.spec_type)
                         + " dst=" + KVCacheSpecTypeToString(dg.spec_type));
    }

    if (sg.spec_type == KVCacheSpecType::LinearAttention) {
        // Asymmetric routes use explicit per-history-row Q/K/V subspans.
        // Legacy LINEAR layouts continue to require equal head shards.
        if (sg.linear_history_rows != 0 || dg.linear_history_rows != 0) {
            if (!compatibleLinearQkvLayouts(src, dst, tag)) {
                return planError("incompatible LINEAR_QKV layout for tag '" + tag + "'");
            }
        } else if (src.headShardCount(tag) != dst.headShardCount(tag)) {
            return planError("LinearAttention asymmetric head sharding is unsupported for tag '" + tag + "'");
        }
    }
    // The side with fewer shards must partition each larger block. Even an
    // MHA spec uses whole-block conversion in a model configured for MLA.
    if (sg.spec_type == KVCacheSpecType::MultiHeadAttention && (src.use_mla || dst.use_mla)
        && src.headShardCount(tag) != dst.headShardCount(tag)) {
        return planError("asymmetric MHA head sharding requires block partitioning for tag '" + tag + "'");
    }

    // (a) 字节维度：还原出的全局尺寸必须一致。**守恒量取决于 slice 模式**：
    //   PAYLOAD_BYTES -> sliceBlockForPeer 以 k_block_payload_bytes 为分母，守恒量是 payload
    //   其它（含 EQUAL_BYTES / NONE） -> 守恒量是 block stride
    const bool payload_sliced = src.effectiveSlice(tag) == CpBlockSliceMode::PAYLOAD_BYTES
                                || dst.effectiveSlice(tag) == CpBlockSliceMode::PAYLOAD_BYTES;
    if (payload_sliced) {
        auto restorePayload = [&tag](const ShardLayout& side) {
            const size_t payload = side.group(tag).k_block_payload_bytes;
            return side.group(tag).pre_sliced ? payload * static_cast<size_t>(side.cpSize()) : payload;
        };
        const size_t src_payload = restorePayload(src);
        const size_t dst_payload = restorePayload(dst);
        if (src_payload != dst_payload) {
            std::ostringstream oss;
            oss << "global payload bytes mismatch for tag '" << tag << "': src=" << src_payload
                << " dst=" << dst_payload;
            return planError(oss.str());
        }
        // 预切片一侧发送的是它的整个 block stride（createBasicBlockInfo 用 stride 而非 payload），
        // 而目的端只为它留出 payload/cp_size 的落点。若预切片侧带对齐填充，两者必然不等。
        for (bool is_src : {true, false}) {
            const ShardLayout& side = is_src ? src : dst;
            const auto&        g    = side.group(tag);
            if (g.pre_sliced && g.kv_block_stride_bytes != g.k_block_payload_bytes) {
                std::ostringstream oss;
                oss << "tag '" << tag << "' on " << sideName(is_src)
                    << " side is PAYLOAD_BYTES pre-sliced but its stride (" << g.kv_block_stride_bytes
                    << ") differs from its payload (" << g.k_block_payload_bytes
                    << "); the padding tail would have no destination";
                return planError(oss.str());
            }
        }
    } else {
        const size_t src_global = src.effectiveGlobalBlockBytes(tag);
        const size_t dst_global = dst.effectiveGlobalBlockBytes(tag);
        if (src_global != dst_global) {
            std::ostringstream oss;
            oss << "global block bytes mismatch for tag '" << tag << "': src=" << src_global
                << " dst=" << dst_global;
            return planError(oss.str());
        }
    }

    if (subBlockCount(src, tag) != subBlockCount(dst, tag)) {
        return planError("sub block count mismatch for tag '" + tag + "'");
    }

    const int src_cp = src.routeCpSize(tag);
    const int dst_cp = dst.routeCpSize(tag);
    if (dst_cp != 1 && dst_cp != src_cp && !supportsAsymmetricPageRR(src, dst, tag)) {
        std::ostringstream oss;
        oss << "unsupported CP topology for tag '" << tag << "': src cp_size=" << src_cp << " dst cp_size=" << dst_cp
            << "; dst must be unsharded, match src, or use divisible local FULL PageRR widths";
        return planError(oss.str());
    }

    // (b) 键维度：单侧自完备性。head 分片与 block RR 分片不能同时作用于同一 group，
    // 否则 rank r 只持有 (head r) x (block ≡ r mod N)，缺 (N-1)/N 的数据。
    // 注意 (a) 抓不到这条 —— 两侧还原出的全局块大小相等，缺失在键维度。
    // (c) cp_slice 的前置条件。CPSlotMapper::sliceBlockForPeer 要求 parts.size() == 1，
    // 即被切的 block 必须是单一 BlockInfo —— 这与 head 维切分（产出 K/V 两个子块）互斥。
    // PAYLOAD_BYTES 的整除检查在生成实际切片的 route 时执行。
    for (bool is_src : {true, false}) {
        const ShardLayout& side  = is_src ? src : dst;
        const auto         slice = side.effectiveSlice(tag);
        if (slice == CpBlockSliceMode::NONE) {
            continue;
        }
        if (side.headShardCount(tag) > 1) {
            std::ostringstream oss;
            oss << "tag '" << tag << "' on " << sideName(is_src) << " side combines cp_slice with head sharding "
                << "(head_shard_count=" << side.headShardCount(tag)
                << "); sliceBlockForPeer requires a single block part";
            return planError(oss.str());
        }
    }

    for (bool is_src : {true, false}) {
        const ShardLayout& side = is_src ? src : dst;
        if (side.effectiveMapping(tag) != CpBlockMappingMode::BLOCK_ROUND_ROBIN) {
            continue;
        }
        const auto& policy = side.group(tag).policy;
        if (policy.group_type != CacheGroupType::FULL) {
            return planError("block round-robin CP requires a FULL P2P group for tag '" + tag + "' on "
                             + sideName(is_src) + " side");
        }
        // Prefill applies this tail globally before CP ownership filtering.
        // Route resolution applies it separately to each owner's keys.
        if (is_src && side.use_group_cache_transfer_policy && policy.active_tail_blocks > 0) {
            return planError("FULL round-robin tag '" + tag
                             + "' cannot use active_tail_blocks in a hybrid Prefill layout");
        }
        if (side.headShardCount(tag) > 1) {
            std::ostringstream oss;
            oss << "tag '" << tag << "' is double-sharded on " << sideName(is_src)
                << " side: head_shard_count=" << side.headShardCount(tag) << " and block round-robin cp_size="
                << side.cpSize() << "; rank r only holds (head r) x (block == r mod cp_size), "
                << "so most (head, block) pairs are held by nobody";
            return planError(oss.str());
        }
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo KVCacheTransferPlanner::buildHeadPairs(const ShardLayout&     src,
                                                 const ShardLayout&     dst,
                                                 const std::string&     tag,
                                                 std::vector<HeadPair>& out) {
    out.clear();
    const int sh = src.headShardCount(tag);
    const int dh = dst.headShardCount(tag);
    if (sh <= 0 || dh <= 0) {
        return planError("invalid head shard count for tag '" + tag + "'");
    }

    if (sh == dh) {
        for (int h = 0; h < sh; ++h) {
            out.push_back(HeadPair{h, h, PartitionSpec{1, 0}, PartitionSpec{1, 0}});
        }
        return ErrorInfo::OkStatus();
    }

    if (sh > dh) {
        if (sh % dh != 0) {
            std::ostringstream oss;
            oss << "head shard counts not divisible for tag '" << tag << "': src=" << sh << " dst=" << dh;
            return planError(oss.str());
        }
        const int n = sh / dh;
        for (int s = 0; s < sh; ++s) {
            // 源端发整块，由目的端按 head 维切分决定落点（修正 §2.3）。
            out.push_back(HeadPair{s, s / n, PartitionSpec{1, 0}, PartitionSpec{n, s % n}});
        }
        return ErrorInfo::OkStatus();
    }

    if (dh % sh != 0) {
        std::ostringstream oss;
        oss << "head shard counts not divisible for tag '" << tag << "': src=" << sh << " dst=" << dh;
        return planError(oss.str());
    }
    const int n = dh / sh;
    for (int d = 0; d < dh; ++d) {
        out.push_back(HeadPair{d / n, d, PartitionSpec{n, d % n}, PartitionSpec{1, 0}});
    }
    return ErrorInfo::OkStatus();
}

bool KVCacheTransferPlanner::collapseResidues(const std::vector<int>& residues,
                                              int                     modulus,
                                              int&                    out_modulus,
                                              int&                    out_residue) {
    if (residues.empty() || modulus <= 0) {
        return false;
    }
    std::set<int> unique(residues.begin(), residues.end());
    const int     first = *unique.begin();

    for (int d = 1; d <= modulus; ++d) {
        if (modulus % d != 0) {
            continue;
        }
        const int     target_residue = first % d;
        std::set<int> expected;
        for (int x = target_residue; x < modulus; x += d) {
            expected.insert(x);
        }
        if (expected == unique) {
            out_modulus = d;
            out_residue = target_residue;
            return true;
        }
    }
    return false;
}

ErrorInfo KVCacheTransferPlanner::buildCpPairs(const ShardLayout&   src,
                                               const ShardLayout&   dst,
                                               const std::string&   tag,
                                               std::vector<CpPair>& out) {
    out.clear();
    const int src_cp_size = src.routeCpSize(tag);
    const int dst_cp_size = dst.routeCpSize(tag);
    if (src_cp_size <= 0 || dst_cp_size <= 0) {
        return planError("invalid cp size for tag '" + tag + "'");
    }
    if (src_cp_size != dst_cp_size && dst_cp_size > 1 && supportsAsymmetricPageRR(src, dst, tag)) {
        // Divisible widths keep the joint period bounded by the larger side.
        const int modulus = std::max(src_cp_size, dst_cp_size);
        for (int residue = 0; residue < modulus; ++residue) {
            CpPair pair;
            pair.src_cp       = residue % src_cp_size;
            pair.dst_cp       = residue % dst_cp_size;
            pair.keys.modulus = modulus;
            pair.keys.residue = residue;
            pair.filter_prefill_keys = src_cp_size < dst_cp_size;
            out.push_back(pair);
        }
        return ErrorInfo::OkStatus();
    }

    // FULL key ownership follows CP; state/SWA endpoint spacing follows group span.
    const bool is_full_group = src.group(tag).policy.group_type == CacheGroupType::FULL;
    // Transport CP controls ranks/byte slices; state/SWA key spacing follows
    // the physical group's token coverage, including on a single-rank Decode.
    const int  modulus    = is_full_group ?
                                src_cp_size :
                                static_cast<int>(src.group(tag).seq_size_per_block / src.cache_key_tokens_per_block);
    const bool src_sliced = src.effectiveSlice(tag) != CpBlockSliceMode::NONE;
    const bool is_compact = !is_full_group && modulus > 1;
    // active_tail_blocks 由编排层折进规则，两侧共用同一个值（Step 1 已校验相等）。
    const int tail_count = static_cast<int>(src.group(tag).policy.active_tail_blocks);

    // (src_cp, dst_cp, slice_index) -> 该 route 覆盖的剩余值集合
    std::map<std::tuple<int, int, int>, std::vector<int>> assign;

    for (int dst_cp = 0; dst_cp < dst_cp_size; ++dst_cp) {
        for (int pos = 0; pos < modulus; ++pos) {
            if (!ownsPosition(dst, tag, static_cast<size_t>(pos), static_cast<size_t>(modulus), dst_cp)) {
                continue;
            }
            std::vector<int> providers;
            for (int src_cp = 0; src_cp < src_cp_size; ++src_cp) {
                if (ownsPosition(src, tag, static_cast<size_t>(pos), static_cast<size_t>(modulus), src_cp)) {
                    providers.push_back(src_cp);
                }
            }
            if (providers.empty()) {
                // FULL group 的 owner 并集必须覆盖全部位置；缺口意味着配置错误。
                // 非 FULL group（COMPACT/SWA）本就只在部分位置有 cache 条目，跳过。
                if (is_full_group) {
                    std::ostringstream oss;
                    oss << "tag '" << tag << "' has no provider for logical position residue " << pos
                        << " (mod " << modulus << ") needed by dst cp_rank " << dst_cp;
                    return planError(oss.str());
                }
                continue;
            }
            if (src_sliced) {
                // 各 src_cp 持有互不相同的字节切片 -> 不选举，全取。
                for (int src_cp : providers) {
                    assign[{src_cp, dst_cp, src_cp}].push_back(pos);
                }
            } else {
                // 各 provider 持有相同字节 -> 选举 cp_rank 最小者。
                assign[{providers.front(), dst_cp, 0}].push_back(pos);
            }
        }
    }

    // COMPACT 的尾键只能挂在唯一一条 route 上，否则会与其它 route 的键集重叠。
    std::map<std::pair<int, int>, int> final_key_route_count;  // (dst_cp, slice_index) -> route count

    for (const auto& [key, residues] : assign) {
        const auto [src_cp, dst_cp, slice_index] = key;
        CpPair pair;
        pair.src_cp      = src_cp;
        pair.dst_cp      = dst_cp;
        pair.slice_index = slice_index;

        int collapsed_modulus = modulus;
        int collapsed_residue = residues.front();
        if (collapseResidues(residues, modulus, collapsed_modulus, collapsed_residue)) {
            pair.keys.modulus = collapsed_modulus;
            pair.keys.residue = collapsed_residue;
            pair.keys.include_final_key = is_compact;
            pair.keys.tail_count        = tail_count;
            if (is_compact) {
                ++final_key_route_count[{dst_cp, slice_index}];
            }
            out.push_back(pair);
        } else {
            // 无法用单一剩余类表达：逐剩余值产出 route（正确但更多 route）。
            std::set<int> unique(residues.begin(), residues.end());
            bool          first = true;
            for (int r : unique) {
                CpPair per_residue      = pair;
                per_residue.keys.modulus = modulus;
                per_residue.keys.residue = r;
                per_residue.keys.include_final_key = is_compact && first;
                per_residue.keys.tail_count         = tail_count;
                if (per_residue.keys.include_final_key) {
                    ++final_key_route_count[{dst_cp, slice_index}];
                }
                first = false;
                out.push_back(per_residue);
            }
        }
    }

    for (const auto& [key, count] : final_key_route_count) {
        if (count > 1) {
            std::ostringstream oss;
            oss << "tag '" << tag << "' would attach the COMPACT final key to " << count
                << " routes for dst cp_rank " << key.first
                << ", which would overlap their key sets; this src/dst CP mapping combination is unsupported";
            return planError(oss.str());
        }
    }
    return ErrorInfo::OkStatus();
}

void KVCacheTransferPlanner::assignSlices(const ShardLayout& src,
                                          const ShardLayout& dst,
                                          const std::string& tag,
                                          const CpPair&      cp_pair,
                                          SliceSpec&         src_slice,
                                          SliceSpec&         dst_slice) {
    src_slice = SliceSpec{};
    dst_slice = SliceSpec{};

    const bool src_pre_sliced = src.group(tag).pre_sliced;
    const bool dst_pre_sliced = dst.group(tag).pre_sliced;

    // 切片只施加在「持整块」的那一侧，且用对端的 CP 几何。
    // isPrefillCpSliced 仅对 PREFILL 角色为真：prefill 的 spec 本身已是切片后的小块，
    // legacy sliceCpDestinationForPeer 也只切目的端。
    if (src_pre_sliced && !dst_pre_sliced) {
        const CpBlockSliceMode mode = src.effectiveSlice(tag);
        if (mode != CpBlockSliceMode::NONE) {
            dst_slice = SliceSpec{mode, src.cpSize(), cp_pair.slice_index};
        }
    } else if (!src_pre_sliced && dst_pre_sliced) {
        const CpBlockSliceMode mode = dst.effectiveSlice(tag);
        if (mode != CpBlockSliceMode::NONE) {
            src_slice = SliceSpec{mode, dst.cpSize(), cp_pair.dst_cp};
        }
    }
}

size_t KVCacheTransferPlanner::sourceReplicaOffset(const std::string& unique_key, int source_rank_count) {
    if (source_rank_count <= 1) {
        return 0;
    }
    // FNV-1a specifies the same byte-level result on both hosts; std::hash does not.
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : unique_key) {
        hash = (hash ^ byte) * 1099511628211ULL;
    }
    return hash % static_cast<size_t>(source_rank_count);
}

PlanResult KVCacheTransferPlanner::plan(const ShardLayout&              src,
                                        const ShardLayout&              dst,
                                        const std::vector<std::string>& tags,
                                        size_t                          source_replica_offset) {
    PlanResult result;
    if (src.pc.tp_size > kMaxPlanRankCount || dst.pc.tp_size > kMaxPlanRankCount) {
        result.error = planError("rank count exceeds " + std::to_string(kMaxPlanRankCount));
        return result;
    }
    if (src.rankCount() <= 0 || dst.rankCount() <= 0) {
        result.error = planError("rank count must be positive");
        return result;
    }
    if (tags.empty()) {
        result.error = planError("tag list is empty");
        return result;
    }

    int next_route_id = 0;

    for (const auto& tag : tags) {
        // ---- Step 1 ----
        result.error = validateTag(src, dst, tag);
        if (result.error.hasError()) {
            result.plan.routes.clear();
            return result;
        }

        // ---- Step 4a: head 维配对 ----
        std::vector<HeadPair> head_pairs;
        result.error = buildHeadPairs(src, dst, tag, head_pairs);
        if (result.error.hasError()) {
            result.plan.routes.clear();
            return result;
        }

        // ---- Step 2 / 3: CP 维配对（含复制型 group 的选举）----
        std::vector<CpPair> cp_pairs;
        result.error = buildCpPairs(src, dst, tag, cp_pairs);
        if (result.error.hasError()) {
            result.plan.routes.clear();
            return result;
        }

        // ---- Step 5: 展开到 rank 对，并在副本类内选举唯一源 ----
        //
        // 一个 rank 在某 tag 上的「内容坐标」= (cpRank, headShard)。坐标相同的 rank 互为
        // 字节相同的副本。目的端每个 rank 都必须被喂到（各自独立内存），源端则在副本类内
        // 选举一个。选举按比例散开，与今天 decode_servers[tp_rank / local_partition_count]
        // 的行为一致。
        using Coord = std::pair<int, int>;
        std::map<Coord, std::vector<int>> source_ranks;
        for (int rank = 0; rank < src.rankCount(); ++rank) {
            source_ranks[{src.routeCpRank(rank, tag), src.headShard(rank, tag)}].push_back(rank);
        }
        std::map<Coord, size_t> destination_counts;
        std::vector<size_t>     destination_indices(dst.rankCount());
        for (int rank = 0; rank < dst.rankCount(); ++rank) {
            destination_indices[rank] = destination_counts[{dst.routeCpRank(rank, tag), dst.headShard(rank, tag)}]++;
        }
        std::vector<std::vector<const CpPair*>> cp_by_destination(dst.routeCpSize(tag));
        for (const auto& pair : cp_pairs) {
            cp_by_destination[pair.dst_cp].push_back(&pair);
        }
        std::vector<std::vector<const HeadPair*>> heads_by_destination(dst.headShardCount(tag));
        for (const auto& pair : head_pairs) {
            heads_by_destination[pair.dst_head].push_back(&pair);
        }

        for (int dst_rank = 0; dst_rank < dst.rankCount(); ++dst_rank) {
            const int    dst_cp         = dst.routeCpRank(dst_rank, tag);
            const int    dst_head       = dst.headShard(dst_rank, tag);
            const size_t dst_index      = destination_indices[dst_rank];
            const size_t dst_class_size = destination_counts.at({dst_cp, dst_head});

            for (const auto* cp : cp_by_destination[dst_cp]) {
                const auto& cp_pair = *cp;
                for (const auto* head : heads_by_destination[dst_head]) {
                    const auto& head_pair = *head;
                    const auto  src_coord = source_ranks.find({cp_pair.src_cp, head_pair.src_head});
                    if (src_coord == source_ranks.end()) {
                        std::ostringstream oss;
                        oss << "tag '" << tag << "' has no src rank with coord (cp=" << cp_pair.src_cp
                            << ", head=" << head_pair.src_head << ")";
                        result.error = planError(oss.str());
                        result.plan.routes.clear();
                        return result;
                    }
                    const auto& src_class = src_coord->second;
                    // 副本类内按比例选举，把出口散开而非全压在第一个 rank 上。
                    const size_t elected_index =
                        ((dst_index * src_class.size()) / dst_class_size + source_replica_offset % src_class.size())
                        % src_class.size();
                    const int src_rank = src_class[elected_index];

                    TransferRoute route;
                    route.route_id      = next_route_id++;
                    route.src_rank      = src_rank;
                    route.dst_rank      = dst_rank;
                    route.cache_tag     = tag;
                    route.src_keys      = cp_pair.keys;
                    route.filter_prefill_keys = cp_pair.filter_prefill_keys;
                    route.src_partition = head_pair.src_partition;
                    route.dst_partition = head_pair.dst_partition;
                    route.linear_qkv_segments =
                        src.group(tag).linear_history_rows != 0 && src.headShardCount(tag) != dst.headShardCount(tag);
                    assignSlices(src, dst, tag, cp_pair, route.src_slice, route.dst_slice);

                    route.src_bytes = viewBytes(src.group(tag), route.src_partition, route.src_slice);
                    route.dst_bytes = viewBytes(dst.group(tag), route.dst_partition, route.dst_slice);
                    route.cache_key_tokens_per_block = src.cache_key_tokens_per_block;
                    route.group_tokens_per_block     = src.group(tag).seq_size_per_block;

                    if (route.src_bytes != route.dst_bytes || route.src_bytes == 0) {
                        std::ostringstream oss;
                        oss << "shape mismatch for tag '" << tag << "' route src_rank=" << route.src_rank
                            << " dst_rank=" << route.dst_rank << ": src_bytes=" << route.src_bytes
                            << " dst_bytes=" << route.dst_bytes;
                        result.error = planError(oss.str());
                        result.plan.routes.clear();
                        return result;
                    }

                    result.plan.routes.push_back(std::move(route));
                }
            }
        }
    }

    result.error = ErrorInfo::OkStatus();
    return result;
}

namespace {

struct PlanDigest {
    uint64_t hash = 14695981039346656037ULL;

    void mix(uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            hash ^= (value >> (i * 8)) & 0xffULL;
            hash *= 1099511628211ULL;
        }
    }

    void mixString(const std::string& value) {
        mix(value.size());
        for (unsigned char byte : value) {
            mix(byte);
        }
    }
};

ErrorInfo validatePipeline(const PipelineTransferLayout& side, int& main_layers, PlanDigest& digest) {
    if (side.stages.empty() || side.worker_count == 0) {
        return planError("PP layout has no stages or workers");
    }
    digest.mix(side.worker_count);
    digest.mix(side.stages.size());
    std::set<int> used_workers;
    for (size_t s = 0; s < side.stages.size(); ++s) {
        const auto& stage = side.stages[s];
        if (stage.main_layer_begin != main_layers || stage.main_layer_count <= 0
            || stage.layer_tags.size() < static_cast<size_t>(stage.main_layer_count)
            || stage.layer_tags.size() > static_cast<size_t>(std::numeric_limits<int>::max() - main_layers)) {
            return planError("invalid main-layer interval for PP stage " + std::to_string(s));
        }
        main_layers += stage.main_layer_count;
        const bool has_draft = stage.layer_tags.size() > static_cast<size_t>(stage.main_layer_count);
        const auto tp        = stage.main_shards.pc.tp_size;
        if (has_draft && (s + 1 != side.stages.size() || stage.draft_shards.pc.tp_size != tp)) {
            return planError("invalid draft layout for PP stage " + std::to_string(s));
        }
        if (tp <= 0 || tp > kMaxPlanRankCount || stage.worker_indices.size() != static_cast<size_t>(tp)) {
            return planError("worker count does not match TP width for PP stage " + std::to_string(s));
        }
        digest.mix(stage.main_layer_begin);
        digest.mix(stage.main_layer_count);
        digest.mix(stage.worker_indices.size());
        for (int worker : stage.worker_indices) {
            if (worker < 0 || static_cast<size_t>(worker) >= side.worker_count || !used_workers.insert(worker).second) {
                return planError("invalid or duplicate worker index for PP stage " + std::to_string(s));
            }
            digest.mix(worker);
        }
        for (const auto* shards : {&stage.main_shards, &stage.draft_shards}) {
            digest.mix(shards->rankCount());
            digest.mix(shards->pc.get_attn_tp_size());
            digest.mix(shards->cpSize());
            digest.mix(shards->cache_key_tokens_per_block);
            digest.mix(shards->use_mla);
            digest.mix(shards->use_group_cache_transfer_policy);
            digest.mix(shards->groups.size());
            for (const auto& [tag, group] : shards->groups) {
                const int heads = group.head_shard_count;
                if (heads <= 0 || shards->pc.get_attn_tp_size() % heads != 0) {
                    return planError("invalid head shard count for PP stage " + std::to_string(s) + " tag=" + tag);
                }
                digest.mixString(tag);
                digest.mix(static_cast<int>(group.policy.group_type));
                digest.mix(static_cast<int>(group.policy.cp_mapping));
                digest.mix(static_cast<int>(group.policy.cp_slice));
                digest.mix(group.policy.active_tail_blocks);
                digest.mix(group.spec_type);
                digest.mix(group.kv_block_stride_bytes);
                digest.mix(group.kv_scale_stride_bytes);
                digest.mix(group.k_block_payload_bytes);
                digest.mix(group.seq_size_per_block);
                digest.mix(group.pre_sliced);
                digest.mix(group.global_kv_heads);
                digest.mix(heads);
                if (group.linear_history_rows != 0) {
                    digest.mix(group.linear_history_rows);
                    digest.mix(group.linear_q_row_bytes);
                    digest.mix(group.linear_k_row_bytes);
                    digest.mix(group.linear_v_row_bytes);
                }
                digest.mix(shards->physicalCpSize(tag));
            }
        }
        digest.mix(stage.layer_tags.size());
        for (size_t layer = 0; layer < stage.layer_tags.size(); ++layer) {
            const auto& shards =
                layer < static_cast<size_t>(stage.main_layer_count) ? stage.main_shards : stage.draft_shards;
            const auto&                 tags = stage.layer_tags[layer];
            const std::set<std::string> ordered(tags.begin(), tags.end());
            if (ordered.empty() || ordered.size() != tags.size()) {
                return planError("empty or duplicate layer tags for PP stage " + std::to_string(s)
                                 + " local_layer=" + std::to_string(layer));
            }
            digest.mix(ordered.size());
            for (const auto& tag : ordered) {
                if (tag.empty() || !shards.hasGroup(tag)) {
                    return planError("missing layout for PP stage " + std::to_string(s) + " tag=" + tag);
                }
                if (!side.stages.front().main_shards.hasGroup(tag)) {
                    return planError("PP cache tag has no Stage 0 block table: " + tag);
                }
                digest.mixString(tag);
            }
        }
    }
    return used_workers.size() == side.worker_count ? ErrorInfo::OkStatus() : planError("PP worker list is incomplete");
}

}  // namespace

PlanResult KVCacheTransferPlanner::planStages(const PipelineTransferLayout& prefill,
                                              const PipelineTransferLayout& decode,
                                              const std::string&            unique_key) {
    PlanDigest layout_digest;
    int        prefill_main = 0, decode_main = 0;
    auto       error = validatePipeline(prefill, prefill_main, layout_digest);
    if (error.ok()) {
        error = validatePipeline(decode, decode_main, layout_digest);
    }
    if (!error.ok()) {
        return {{}, std::move(error)};
    }
    if (prefill_main != decode_main) {
        return {{}, planError("Prefill and Decode cover different main-model layers")};
    }
    const auto draftCount = [](const auto& stage) {
        return static_cast<int>(stage.layer_tags.size()) - stage.main_layer_count;
    };
    const int draft_layers         = draftCount(decode.stages.back());
    const int prefill_draft_layers = draftCount(prefill.stages.back());
    if (prefill_draft_layers < draft_layers) {
        return {{},
                planError("Prefill is missing draft layers required by Decode: Prefill="
                          + std::to_string(prefill_draft_layers) + " Decode=" + std::to_string(draft_layers))};
    }

    TransferPlan result;
    result.layout_digest = layout_digest.hash;
    for (size_t s = 0, d = 0; s < prefill.stages.size() && d < decode.stages.size();) {
        const auto& src     = prefill.stages[s];
        const auto& dst     = decode.stages[d];
        const int   src_end = src.main_layer_begin + src.main_layer_count;
        const int   dst_end = dst.main_layer_begin + dst.main_layer_count;
        const auto  append_routes =
            [&](int begin, int end, const ShardLayout& src_shards, const ShardLayout& dst_shards) -> ErrorInfo {
            // Order by the exact layer set, then by tag, then by the TP planner's route order.
            std::map<std::string, std::vector<int>> layers_by_tag;
            for (int layer = begin; layer < end; ++layer) {
                const auto& src_tags = src.layer_tags[layer - src.main_layer_begin];
                for (const auto& tag : dst.layer_tags[layer - dst.main_layer_begin]) {
                    if (std::find(src_tags.begin(), src_tags.end(), tag) == src_tags.end()) {
                        return planError("Prefill is missing global layer " + std::to_string(layer) + " tag=" + tag);
                    }
                    layers_by_tag[tag].push_back(layer);
                }
            }
            std::vector<std::pair<std::vector<int>, std::string>> groups;
            for (auto& [tag, layers] : layers_by_tag) {
                groups.emplace_back(std::move(layers), tag);
            }
            std::sort(groups.begin(), groups.end());
            const auto offset = sourceReplicaOffset(unique_key, src_shards.rankCount());
            for (const auto& [layers, tag] : groups) {
                auto shards = plan(src_shards, dst_shards, {tag}, offset);
                if (!shards.ok()) {
                    return shards.error;
                }
                std::set<int> covered_lanes;
                for (auto& route : shards.plan.routes) {
                    covered_lanes.insert(route.dst_rank);
                    const size_t src_scale = src_shards.group(tag).kv_scale_stride_bytes;
                    const size_t dst_scale = dst_shards.group(tag).kv_scale_stride_bytes;
                    const auto   src_parts = static_cast<size_t>(route.src_partition.count);
                    const auto   dst_parts = static_cast<size_t>(route.dst_partition.count);
                    if (src_scale % src_parts != 0 || dst_scale % dst_parts != 0
                        || src_scale / src_parts != dst_scale / dst_parts
                        || (src_scale != 0 && (route.src_slice.count > 1 || route.dst_slice.count > 1))) {
                        return planError("incompatible scale layout for PP transfer tag=" + tag);
                    }
                    if (result.routes.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) {
                        return planError("PP transfer has too many routes");
                    }
                    route.route_id = static_cast<int>(result.routes.size());
                    route.src_rank = src.worker_indices[route.src_rank];
                    route.dst_rank = dst.worker_indices[route.dst_rank];
                    route.src_slice_payload_bytes = src_shards.group(tag).k_block_payload_bytes;
                    route.dst_slice_payload_bytes = dst_shards.group(tag).k_block_payload_bytes;
                    for (int layer : layers) {
                        route.layers.push_back({layer, layer - src.main_layer_begin, layer - dst.main_layer_begin});
                    }
                    result.routes.push_back(std::move(route));
                }
                if (covered_lanes.size() != dst.worker_indices.size()) {
                    return planError("incomplete PP route coverage for global layer " + std::to_string(layers.front())
                                     + " tag=" + tag);
                }
            }
            return ErrorInfo::OkStatus();
        };
        error = append_routes(std::max(src.main_layer_begin, dst.main_layer_begin),
                              std::min(src_end, dst_end),
                              src.main_shards,
                              dst.main_shards);
        if (error.ok() && src_end == prefill_main && dst_end == decode_main && draft_layers > 0) {
            error = append_routes(prefill_main, prefill_main + draft_layers, src.draft_shards, dst.draft_shards);
        }
        if (!error.ok()) {
            return {{}, std::move(error)};
        }
        s += src_end <= dst_end;
        d += dst_end <= src_end;
    }
    return {std::move(result), ErrorInfo::OkStatus()};
}

uint64_t TransferPlan::digest() const {
    PlanDigest digest;
    const auto mix = [&digest](uint64_t value) { digest.mix(value); };
    if (layout_digest != 0) {
        mix(layout_digest);
    }
    for (const auto& r : routes) {
        mix(static_cast<uint64_t>(r.route_id));
        mix(static_cast<uint64_t>(r.src_rank));
        mix(static_cast<uint64_t>(r.dst_rank));
        if (!r.layers.empty()) {
            mix(r.layers.size());
            for (const auto& layer : r.layers) {
                mix(layer.global_id);
                mix(layer.src_local_id);
                mix(layer.dst_local_id);
            }
            mix(r.src_slice_payload_bytes);
            mix(r.dst_slice_payload_bytes);
        }
        for (char c : r.cache_tag) {
            mix(static_cast<uint64_t>(static_cast<unsigned char>(c)));
        }
        mix(static_cast<uint64_t>(r.src_keys.modulus));
        mix(static_cast<uint64_t>(r.src_keys.residue));
        mix(static_cast<uint64_t>(r.src_keys.include_final_key ? 1 : 0));
        mix(static_cast<uint64_t>(r.src_keys.tail_count));
        mix(static_cast<uint64_t>(r.src_keys.replica_split_count));
        mix(static_cast<uint64_t>(r.src_keys.replica_split_index));
        mix(static_cast<uint64_t>(r.src_partition.count));
        mix(static_cast<uint64_t>(r.src_partition.id));
        mix(static_cast<uint64_t>(r.dst_partition.count));
        mix(static_cast<uint64_t>(r.dst_partition.id));
        mix(static_cast<uint64_t>(r.src_slice.mode));
        mix(static_cast<uint64_t>(r.src_slice.count));
        mix(static_cast<uint64_t>(r.src_slice.index));
        mix(static_cast<uint64_t>(r.dst_slice.mode));
        mix(static_cast<uint64_t>(r.dst_slice.count));
        mix(static_cast<uint64_t>(r.dst_slice.index));
        mix(static_cast<uint64_t>(r.src_bytes));
        mix(static_cast<uint64_t>(r.dst_bytes));
        mix(static_cast<uint64_t>(r.cache_key_tokens_per_block));
        mix(static_cast<uint64_t>(r.group_tokens_per_block));
        if (r.linear_qkv_segments) {
            mix(1);
        }
    }
    return digest.hash;
}

ShardLayout ShardLayout::forPeer(const ShardLayout&       self,
                                 const ParallelismConfig& peer_pc,
                                 bool                     peer_is_prefill_role) {
    ShardLayout peer;
    peer.pc                              = peer_pc;
    peer.cache_key_tokens_per_block      = self.cache_key_tokens_per_block;
    peer.use_group_cache_transfer_policy = self.use_group_cache_transfer_policy;
    peer.use_mla                         = self.use_mla;
    peer.groups                          = self.groups;  // same-build: policy / spec 类型 / seq 尺寸两侧相同

    peer.deriveHeadShardCounts();

    for (auto& [tag, g] : peer.groups) {
        // pre_sliced 是角色属性：仅 PREFILL 角色的 cp_slice group 的 spec 本身被切片。
        const bool slice_active = peer.cpSize() > 1 && g.policy.group_type != CacheGroupType::FULL
                                  && g.policy.cp_slice != CpBlockSliceMode::NONE;
        g.pre_sliced = peer_is_prefill_role && slice_active;

        // 还原全局再除对端因子；等价于以替换后的 pc 调 localKvHeadNumForSpec。
        const size_t global_kv    = self.effectiveGlobalBlockBytes(tag);
        size_t       divisor      = static_cast<size_t>(std::max(1, peer.headShardCount(tag)));
        if (g.pre_sliced) {
            divisor *= static_cast<size_t>(std::max(1, peer.cpSize()));
        }
        g.kv_block_stride_bytes = global_kv / divisor;

        // PAYLOAD_BYTES changes the spec's payload as well as its stride.
        // EQUAL_BYTES only slices the stride; its spec still describes the full
        // payload, so it must not use the stride's CP divisor here.
        size_t global_payload = self.group(tag).k_block_payload_bytes
                                * static_cast<size_t>(std::max(1, self.headShardCount(tag)));
        if (self.group(tag).pre_sliced && self.effectiveSlice(tag) == CpBlockSliceMode::PAYLOAD_BYTES) {
            global_payload *= static_cast<size_t>(self.cpSize());
        }
        size_t payload_divisor = static_cast<size_t>(std::max(1, peer.headShardCount(tag)));
        if (g.pre_sliced && peer.effectiveSlice(tag) == CpBlockSliceMode::PAYLOAD_BYTES) {
            payload_divisor *= static_cast<size_t>(peer.cpSize());
        }
        g.k_block_payload_bytes = global_payload / payload_divisor;

        if (self.group(tag).kv_scale_stride_bytes > 0) {
            const size_t self_divisor = static_cast<size_t>(std::max(1, self.headShardCount(tag)))
                                        * (self.group(tag).pre_sliced ?
                                               static_cast<size_t>(std::max(1, self.cpSize())) :
                                               1);
            const size_t global_scale = self.group(tag).kv_scale_stride_bytes * self_divisor;
            g.kv_scale_stride_bytes    = global_scale / divisor;
        }
    }
    return peer;
}

}  // namespace rtp_llm
