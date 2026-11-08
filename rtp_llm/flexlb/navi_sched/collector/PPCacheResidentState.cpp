#include "aios/navi_sched/collector/PPCacheResidentState.h"
#include <rapidjson/document.h>
#include <algorithm>
#include <stdexcept>

namespace navi_sched {
namespace {
int hexDigit(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
std::string decodeHash(const rapidjson::Value &value)
{
    if (!value.IsString()) throw std::invalid_argument("invalid V6D block hash");
    std::string bytes;
    int high = -1;
    for (size_t i = 0; i < value.GetStringLength(); ++i) {
        const char c = value.GetString()[i];
        const bool space =
            c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
        if (space && high < 0) continue;
        const int digit = hexDigit(c);
        if (digit < 0) throw std::invalid_argument("invalid V6D block hash");
        if (high < 0)
            high = digit;
        else {
            bytes.push_back(char((high << 4) | digit));
            high = -1;
        }
    }
    if (high >= 0) throw std::invalid_argument("invalid V6D block hash");
    return bytes;
}
uint64_t wireHash(const std::string &bytes, size_t offset)
{
    uint64_t hash = 0;
    for (size_t i = offset + 24; i < offset + 32; ++i)
        hash = (hash << 8) | static_cast<unsigned char>(bytes[i]);
    return hash;
}
}

PPCacheResidentState::PPCacheResidentState(std::vector<PPCacheGroupRole> roles,
                                           std::vector<uint32_t> blockScales)
    : _roles(std::move(roles))
    , _blockScales(std::move(blockScales))
    , _counts(_roles.size())
{
    if (_roles.empty() || _roles.size() != _blockScales.size())
        throw std::invalid_argument("invalid PP cache group policy");
    for (size_t i = 0; i < _roles.size(); ++i)
        if (!_blockScales[i] ||
            (_roles[i] != PPCacheGroupRole::FULL && _roles[i] != PPCacheGroupRole::TAIL))
            throw std::invalid_argument("invalid PP cache group policy");
}

PPCacheResidentState::Metadata PPCacheResidentState::parse(
    const std::string &identity) const
{
    rapidjson::Document document;
    document.Parse(identity.data(), identity.size());
    if (document.HasParseError() || !document.IsObject() ||
        !document.HasMember("group_id") || !document["group_id"].IsUint() ||
        document["group_id"].GetUint() >= _roles.size() ||
        !document.HasMember("block_hashes") || !document["block_hashes"].IsArray() ||
        document["block_hashes"].Empty())
        throw std::invalid_argument("invalid PP cache identity");
    Metadata metadata;
    metadata.identity = identity;
    metadata.group = document["group_id"].GetUint();
    for (const auto &value : document["block_hashes"].GetArray()) {
        const auto bytes = decodeHash(value);
        if (bytes.size() != 32 &&
            bytes.size() != uint64_t(32) * _blockScales[metadata.group])
            throw std::invalid_argument("invalid PP cache group hash size");
        if (_roles[metadata.group] == PPCacheGroupRole::TAIL)
            metadata.hashes.push_back(wireHash(bytes, bytes.size() - 32));
        else
            for (size_t offset = 0; offset < bytes.size(); offset += 32)
                metadata.hashes.push_back(wireHash(bytes, offset));
    }
    return metadata;
}

bool PPCacheResidentState::ready(uint64_t hash, PPCacheGroupRole role) const
{
    bool found = false;
    for (size_t group = 0; group < _roles.size(); ++group) {
        if (_roles[group] != role) continue;
        found = true;
        if (!_counts[group].count(hash)) return false;
    }
    return found;
}

void PPCacheResidentState::add(const Metadata &metadata)
{
    for (auto hash : metadata.hashes) ++_counts[metadata.group][hash];
}
void PPCacheResidentState::erase(const Metadata &metadata)
{
    for (auto hash : metadata.hashes) {
        auto found = _counts[metadata.group].find(hash);
        if (!--found->second) _counts[metadata.group].erase(found);
    }
}

PPCacheShardUpdate PPCacheResidentState::change(const std::string &id,
                                                const Metadata *replacement)
{
    auto previous = _objects.find(id);
    std::unordered_set<uint64_t> affected;
    if (previous != _objects.end())
        affected.insert(previous->second.hashes.begin(), previous->second.hashes.end());
    if (replacement)
        affected.insert(replacement->hashes.begin(), replacement->hashes.end());
    std::unordered_map<uint64_t, std::pair<bool, bool>> before;
    for (auto hash : affected)
        before[hash] = {ready(hash, PPCacheGroupRole::FULL),
                        ready(hash, PPCacheGroupRole::TAIL)};
    if (previous != _objects.end()) {
        erase(previous->second);
        _objects.erase(previous);
    }
    if (replacement) {
        add(*replacement);
        _objects.emplace(id, *replacement);
    }
    PPCacheShardUpdate delta;
    for (auto hash : affected) {
        const bool full = ready(hash, PPCacheGroupRole::FULL);
        const bool tail = ready(hash, PPCacheGroupRole::TAIL);
        if (before[hash].first != full)
            (full ? delta.addedFull : delta.removedFull).push_back(hash);
        if (before[hash].second != tail)
            (tail ? delta.addedTail : delta.removedTail).push_back(hash);
    }
    return delta;
}

PPCacheShardUpdate PPCacheResidentState::resident(const PPCacheResidentObject &object)
{
    if (object.objectId.empty()) throw std::invalid_argument("empty PP cache object id");
    const auto existing = _objects.find(object.objectId);
    if (existing != _objects.end() && existing->second.identity == object.identity)
        return {};
    if (object.identity.empty()) return change(object.objectId, nullptr);
    const auto metadata = parse(object.identity);
    return change(object.objectId, &metadata);
}
PPCacheShardUpdate PPCacheResidentState::removed(const std::string &objectId)
{
    if (objectId.empty()) throw std::invalid_argument("empty PP cache object id");
    return change(objectId, nullptr);
}
PPCacheShardUpdate PPCacheResidentState::replace(
    const std::vector<PPCacheResidentObject> &objects)
{
    PPCacheResidentState replacement(_roles, _blockScales);
    std::unordered_set<std::string> ids;
    for (const auto &object : objects) {
        if (!ids.insert(object.objectId).second)
            throw std::invalid_argument("duplicate PP cache object id");
        replacement.resident(object);
    }
    PPCacheShardUpdate snapshot;
    snapshot.fullSnapshot = true;
    for (size_t group = 0; group < _roles.size(); ++group)
        if (_roles[group] == PPCacheGroupRole::FULL) {
            for (const auto &entry : replacement._counts[group])
                if (replacement.ready(entry.first, PPCacheGroupRole::FULL))
                    snapshot.addedFull.push_back(entry.first);
            break;
        }
    for (size_t group = 0; group < _roles.size(); ++group)
        if (_roles[group] == PPCacheGroupRole::TAIL) {
            for (const auto &entry : replacement._counts[group])
                if (replacement.ready(entry.first, PPCacheGroupRole::TAIL))
                    snapshot.addedTail.push_back(entry.first);
            break;
        }
    _counts = std::move(replacement._counts);
    _objects = std::move(replacement._objects);
    return snapshot;
}

}
