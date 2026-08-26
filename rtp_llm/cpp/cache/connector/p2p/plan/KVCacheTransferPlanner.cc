#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"

#include "rtp_llm/cpp/cache/CPSlotMapper.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

namespace rtp_llm {

namespace {

ErrorInfo planError(const std::string& msg) {
    return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, "transfer plan: " + msg);
}

std::string sideName(bool is_src) {
    return is_src ? "src" : "dst";
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
        return group.k_block_payload_bytes / partition_count / slice_count;
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
    const int cp_size = side.cpSize();
    if (cp_size <= 0 || cp_rank < 0 || cp_rank >= cp_size) {
        return false;
    }
    CacheGroupPolicy policy = side.group(tag).policy;
    // 镜像 CPSlotMapper::layoutForGroup：cpSize() <= 1 时 mapping 退化为 NONE。
    policy.cp_mapping = side.effectiveMapping(tag);
    return CPSlotMapper::physicalBlockPosition(policy, pos, count, cp_rank, cp_size).has_value();
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

    // (b0) CP 形态白名单：只允许 dst 不分片（prefill CP N -> decode CP 1）或两侧完全相等
    // （今天 §2.1 唯一允许的对称形态）。两侧都分片且不相等（如 p cp=2 -> d cp=4）不予支持：
    // 那需要「模 lcm(src,dst) 的剩余类 + CRT 求 residue」整套机制，而 vLLM
    // (_validate_remote_parallel_config) 与 SGLang (common/conn.py 的 assert) 同样禁止
    // 两侧 CP 并存。挡在这里之后 modulus 恒等于 src.cpSize()，residue 恒等于 src_cp。
    if (dst.cpSize() != 1 && dst.cpSize() != src.cpSize()) {
        std::ostringstream oss;
        oss << "unsupported CP topology for tag '" << tag << "': src cp_size=" << src.cpSize()
            << " dst cp_size=" << dst.cpSize()
            << "; dst must be either unsharded (1) or exactly match src";
        return planError(oss.str());
    }

    // (b) 键维度：单侧自完备性。head 分片与 block RR 分片不能同时作用于同一 group，
    // 否则 rank r 只持有 (head r) x (block ≡ r mod N)，缺 (N-1)/N 的数据。
    // 注意 (a) 抓不到这条 —— 两侧还原出的全局块大小相等，缺失在键维度。
    // (c) cp_slice 的前置条件。CPSlotMapper::sliceBlockForPeer 要求 parts.size() == 1，
    // 即被切的 block 必须是单一 BlockInfo —— 这与 head 维切分（产出 K/V 两个子块）互斥。
    // 同时 PAYLOAD_BYTES 模式以 k_block_payload_bytes 为分母，须为正且能被 cp_size 整除。
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
        if (slice == CpBlockSliceMode::PAYLOAD_BYTES) {
            const size_t payload = side.group(tag).k_block_payload_bytes;
            if (payload == 0 || payload % static_cast<size_t>(side.cpSize()) != 0) {
                std::ostringstream oss;
                oss << "tag '" << tag << "' on " << sideName(is_src)
                    << " side uses PAYLOAD_BYTES cp_slice but k_block_payload_bytes=" << payload
                    << " is not a positive multiple of cp_size=" << side.cpSize();
                return planError(oss.str());
            }
        }
    }

    for (bool is_src : {true, false}) {
        const ShardLayout& side = is_src ? src : dst;
        if (side.headShardCount(tag) > 1 && side.cpSize() > 1
            && side.effectiveMapping(tag) == CpBlockMappingMode::BLOCK_ROUND_ROBIN) {
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
    const int src_cp_size = src.cpSize();
    const int dst_cp_size = dst.cpSize();
    if (src_cp_size <= 0 || dst_cp_size <= 0) {
        return planError("invalid cp size for tag '" + tag + "'");
    }

    // Step 1 的 (b0) 已保证 dst_cp_size ∈ {1, src_cp_size}，故 lcm(src, dst) 恒等于 src_cp_size，
    // 无需 lcm / CRT。
    const int  modulus    = src_cp_size;
    const bool src_sliced = src.effectiveSlice(tag) != CpBlockSliceMode::NONE;
    const bool is_compact = src.effectiveMapping(tag) == CpBlockMappingMode::COMPACT_LAST_RANK
                            || dst.effectiveMapping(tag) == CpBlockMappingMode::COMPACT_LAST_RANK;
    const bool is_full_group = src.group(tag).policy.group_type == CacheGroupType::FULL;
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

std::vector<int>
KVCacheTransferPlanner::ranksWithCoord(const ShardLayout& side, const std::string& tag, int cp_rank, int head_shard) {
    std::vector<int> out;
    for (int r = 0; r < side.rankCount(); ++r) {
        if (side.cpRank(r) == cp_rank && side.headShard(r, tag) == head_shard) {
            out.push_back(r);
        }
    }
    return out;
}

PlanResult KVCacheTransferPlanner::plan(const ShardLayout&              src,
                                        const ShardLayout&              dst,
                                        const std::vector<std::string>& tags) {
    PlanResult result;
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
        for (int dst_rank = 0; dst_rank < dst.rankCount(); ++dst_rank) {
            const int dst_cp   = dst.cpRank(dst_rank);
            const int dst_head = dst.headShard(dst_rank, tag);

            const std::vector<int> dst_class = ranksWithCoord(dst, tag, dst_cp, dst_head);
            const auto             dst_it    = std::find(dst_class.begin(), dst_class.end(), dst_rank);
            const size_t           dst_index = static_cast<size_t>(std::distance(dst_class.begin(), dst_it));
            const size_t           dst_class_size = std::max<size_t>(1, dst_class.size());

            for (const auto& cp_pair : cp_pairs) {
                if (cp_pair.dst_cp != dst_cp) {
                    continue;
                }
                for (const auto& head_pair : head_pairs) {
                    if (head_pair.dst_head != dst_head) {
                        continue;
                    }

                    const std::vector<int> src_class =
                        ranksWithCoord(src, tag, cp_pair.src_cp, head_pair.src_head);
                    if (src_class.empty()) {
                        std::ostringstream oss;
                        oss << "tag '" << tag << "' has no src rank with coord (cp=" << cp_pair.src_cp
                            << ", head=" << head_pair.src_head << ")";
                        result.error = planError(oss.str());
                        result.plan.routes.clear();
                        return result;
                    }
                    // 副本类内按比例选举，把出口散开而非全压在第一个 rank 上。
                    const size_t elected_index = (dst_index * src_class.size()) / dst_class_size;
                    const int    src_rank       = src_class[std::min(elected_index, src_class.size() - 1)];

                    TransferRoute route;
                    route.route_id      = next_route_id++;
                    route.src_rank      = src_rank;
                    route.dst_rank      = dst_rank;
                    route.cache_tag     = tag;
                    route.src_keys      = cp_pair.keys;
                    route.src_partition = head_pair.src_partition;
                    route.dst_partition = head_pair.dst_partition;
                    assignSlices(src, dst, tag, cp_pair, route.src_slice, route.dst_slice);

                    route.src_bytes = viewBytes(src.group(tag), route.src_partition, route.src_slice);
                    route.dst_bytes = viewBytes(dst.group(tag), route.dst_partition, route.dst_slice);

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

uint64_t TransferPlan::digest() const {
    // FNV-1a over the route fields, in vector order.
    uint64_t       hash  = 1469598103934665603ULL;
    constexpr auto prime = 1099511628211ULL;
    auto           mix   = [&hash](uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            hash ^= (value >> (i * 8)) & 0xffULL;
            hash *= prime;
        }
    };
    for (const auto& r : routes) {
        mix(static_cast<uint64_t>(r.route_id));
        mix(static_cast<uint64_t>(r.src_rank));
        mix(static_cast<uint64_t>(r.dst_rank));
        for (char c : r.cache_tag) {
            mix(static_cast<uint64_t>(static_cast<unsigned char>(c)));
        }
        mix(static_cast<uint64_t>(r.src_keys.modulus));
        mix(static_cast<uint64_t>(r.src_keys.residue));
        mix(static_cast<uint64_t>(r.src_keys.include_final_key ? 1 : 0));
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
    }
    return hash;
}

ShardLayout ShardLayout::forPeer(const ShardLayout&       self,
                                 const ParallelismConfig& peer_pc,
                                 bool                     peer_is_prefill_role) {
    ShardLayout peer;
    peer.pc     = peer_pc;
    peer.groups = self.groups;  // same-build: policy / spec 类型 / seq 尺寸两侧相同

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
