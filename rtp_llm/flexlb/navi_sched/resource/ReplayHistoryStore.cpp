#include "rtp_llm/flexlb/navi_sched/resource/ReplayHistoryStore.h"

#include "rtp_llm/flexlb/navi_sched/data/BlockHashUtil.h"
#include "autil/CRC32C.h"
#include "autil/DataBuffer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <limits>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace navi_sched {
namespace {

constexpr uint32_t RECORD_MAGIC = 0x4E535248; // NSRH
constexpr uint32_t LEGACY_RECORD_VERSION = 1;
constexpr uint32_t RECORD_VERSION = 2;
constexpr int32_t HISTORY_PAYLOAD_VERSION = 1;
constexpr int32_t FIRST_COMPATIBLE_LEGACY_DATA_VERSION = 9;
constexpr int32_t LAST_COMPATIBLE_LEGACY_DATA_VERSION = 10;
constexpr uint64_t HEADER_SIZE = sizeof(uint32_t) * 3 + sizeof(int64_t) +
                                 sizeof(uint64_t);
constexpr uint64_t MAX_PAYLOAD_SIZE = 256ULL * 1024 * 1024;
constexpr char COMPACTION_SUFFIX[] = ".tmp";

class HistoryDirectoryLock {
public:
    explicit HistoryDirectoryLock(const std::filesystem::path &directory)
        : _fd(::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC))
    {
        if (_fd < 0) {
            return;
        }
        int result;
        do {
            result = ::flock(_fd, LOCK_EX);
        } while (result != 0 && errno == EINTR);
        if (result != 0) {
            ::close(_fd);
            _fd = -1;
        }
    }
    ~HistoryDirectoryLock()
    {
        if (_fd >= 0) {
            ::close(_fd);
        }
    }
    HistoryDirectoryLock(const HistoryDirectoryLock &) = delete;
    HistoryDirectoryLock &operator=(const HistoryDirectoryLock &) = delete;
    bool valid() const { return _fd >= 0; }

private:
    int _fd;
};

uint64_t maxPersistedRecordCount(size_t historyLimit)
{
    const auto limit = static_cast<uint64_t>(historyLimit);
    if (limit > std::numeric_limits<uint64_t>::max() / 2) {
        return std::numeric_limits<uint64_t>::max();
    }
    return limit * 2;
}

bool isHistoryKeyDirectory(const std::filesystem::path &directory)
{
    const auto name = directory.filename().string();
    return name.size() == 16 &&
           std::all_of(name.begin(), name.end(), [](unsigned char value) {
               return std::isxdigit(value) != 0;
           });
}

bool migrateLegacyHistory(const std::filesystem::path &directory,
                          const std::filesystem::path &preferredLegacy)
{
    if (preferredLegacy.empty()) {
        return true;
    }
    const auto dataFile = directory / "queries.bin";
    const auto temporary = dataFile.string() + COMPACTION_SUFFIX;
    std::error_code error;
    std::filesystem::path migrationSource;
    const bool dataFileExists = std::filesystem::exists(dataFile, error);
    if (error) {
        return false;
    }
    if (dataFileExists && !std::filesystem::is_regular_file(dataFile, error)) {
        return false;
    }
    if (error) {
        return false;
    }

    std::vector<std::filesystem::path> legacyDirectories;
    std::filesystem::file_time_type newestTime =
        std::filesystem::file_time_type::min();
    for (std::filesystem::directory_iterator iterator(directory, error), end;
         iterator != end; iterator.increment(error)) {
        if (error) {
            return false;
        }
        const auto path = iterator->path();
        if (!iterator->is_directory(error)) {
            if (error) {
                return false;
            }
            continue;
        }
        if (!isHistoryKeyDirectory(path)) {
            continue;
        }
        legacyDirectories.push_back(path);
        const auto candidate = path / "queries.bin";
        const bool candidateExists = std::filesystem::exists(candidate, error);
        if (error) {
            return false;
        }
        if (!candidateExists) {
            continue;
        }
        if (!std::filesystem::is_regular_file(candidate, error) || error) {
            return false;
        }
        const auto modifiedTime = std::filesystem::last_write_time(candidate, error);
        if (error) {
            return false;
        }
        if (migrationSource.empty() || modifiedTime > newestTime ||
            (modifiedTime == newestTime && candidate < migrationSource)) {
            newestTime = modifiedTime;
            migrationSource = candidate;
        }
    }
    if (error) {
        return false;
    }

    if (!dataFileExists) {
        const auto preferredDataFile = preferredLegacy / "queries.bin";
        const bool preferredExists = std::filesystem::exists(preferredDataFile, error);
        if (error) {
            return false;
        }
        if (preferredExists &&
            (!std::filesystem::is_regular_file(preferredDataFile, error) || error)) {
            return false;
        }
        if (preferredExists) {
            migrationSource = preferredDataFile;
        }
        if (!migrationSource.empty()) {
            std::filesystem::rename(migrationSource, dataFile, error);
            if (error) {
                error.clear();
                std::filesystem::copy_file(migrationSource, temporary,
                                           std::filesystem::copy_options::none,
                                           error);
                if (error) {
                    return false;
                }
                std::filesystem::rename(temporary, dataFile, error);
                if (error) {
                    std::error_code ignored;
                    std::filesystem::remove(temporary, ignored);
                    return false;
                }
            }
        }
    }

    for (const auto &path : legacyDirectories) {
        std::filesystem::remove_all(path, error);
        if (error) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool readValue(std::istream &input, T &value)
{
    return static_cast<bool>(
        input.read(reinterpret_cast<char *>(&value), sizeof(value)));
}

template <typename T>
bool writeValue(std::ostream &output, const T &value)
{
    return static_cast<bool>(
        output.write(reinterpret_cast<const char *>(&value), sizeof(value)));
}

bool legacyDataVersionCompatible(int32_t version)
{
    return version >= FIRST_COMPATIBLE_LEGACY_DATA_VERSION &&
           version <= LAST_COMPATIBLE_LEGACY_DATA_VERSION;
}

void serializeHistoryRequest(autil::DataBuffer &buffer,
                             const NaviSchedScheduleRequest &request)
{
    buffer.write(HISTORY_PAYLOAD_VERSION);
    buffer.write(request.requestId);
    buffer.write(request.bizName);
    buffer.write(request.promptTokenCount);
    buffer.write(request.blockSize);
    BlockHashUtil::write(buffer, request.blockHashes);
    buffer.write(request.timeoutMs);
    buffer.write(request.forwardPayload);
}

bool deserializeHistoryRequest(autil::DataBuffer &buffer,
                               NaviSchedScheduleRequest &request)
{
    int32_t version = 0;
    buffer.read(version);
    if (version != HISTORY_PAYLOAD_VERSION) {
        return false;
    }
    request.version = NAVI_SCHED_DATA_VERSION;
    buffer.read(request.requestId);
    buffer.read(request.bizName);
    buffer.read(request.promptTokenCount);
    buffer.read(request.blockSize);
    BlockHashUtil::read(buffer, request.blockHashes);
    buffer.read(request.timeoutMs);
    buffer.read(request.forwardPayload);
    return request.valid() && buffer.getDataLen() == 0;
}

bool deserializeLegacyHistoryRequest(autil::DataBuffer &buffer,
                                     NaviSchedScheduleRequest &request)
{
    int32_t version = 0;
    buffer.read(version);
    if (!legacyDataVersionCompatible(version)) {
        return false;
    }
    request.version = NAVI_SCHED_DATA_VERSION;
    buffer.read(request.requestId);
    buffer.read(request.bizName);
    buffer.read(request.promptTokenCount);
    buffer.read(request.blockSize);
    BlockHashUtil::read(buffer, request.blockHashes);
    buffer.read(request.timeoutMs);
    buffer.read(request.forwardPayload);
    return request.valid() && buffer.getDataLen() == 0;
}

} // namespace

ReplayHistoryStore::~ReplayHistoryStore()
{
    if (_indexedFd >= 0) {
        ::close(_indexedFd);
    }
}

bool ReplayHistoryStore::configure(const std::string &directory,
                                   size_t historyLimit,
                                   const std::string &legacyDirectory)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (_indexedFd >= 0) {
        ::close(_indexedFd);
        _indexedFd = -1;
    }
    _historyLimit = std::max<size_t>(1, historyLimit);
    _index.clear();
    _volatileSamples.clear();
    _validDataSize = 0;
    _persistedRecordCount = 0;
    _incompatibleRecordCount = 0;
    _dataFile.clear();
    if (directory.empty()) {
        _dataFile.clear();
        return true;
    }
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        return false;
    }
    const std::filesystem::path historyDirectory(directory);
    HistoryDirectoryLock diskLock(historyDirectory);
    if (!diskLock.valid()) {
        return false;
    }
    const auto dataFile = historyDirectory / "queries.bin";
    std::filesystem::remove(dataFile.string() + COMPACTION_SUFFIX, error);
    if (error) {
        return false;
    }
    if (!migrateLegacyHistory(historyDirectory,
                              std::filesystem::path(legacyDirectory))) {
        return false;
    }
    _dataFile = dataFile.string();
    const bool dataFileExists = std::filesystem::exists(dataFile, error);
    if (error) {
        _dataFile.clear();
        return false;
    }
    if (!dataFileExists) {
        std::ofstream output(_dataFile, std::ios::binary | std::ios::trunc);
        if (!output) {
            _dataFile.clear();
            return false;
        }
    }
    if (!scanLocked()) {
        _dataFile.clear();
        _index.clear();
        return false;
    }
    if ((_incompatibleRecordCount > 0 ||
         _persistedRecordCount > maxPersistedRecordCount(_historyLimit)) &&
        !compactLocked()) {
        _dataFile.clear();
        _index.clear();
        _validDataSize = 0;
        _persistedRecordCount = 0;
        _incompatibleRecordCount = 0;
        return false;
    }
    return true;
}

bool ReplayHistoryStore::refreshLocked() const
{
    struct stat indexed;
    struct stat current;
    // Keep the indexed inode open so replacement cannot reuse its inode number.
    if (_indexedFd >= 0 && ::fstat(_indexedFd, &indexed) == 0 &&
        ::stat(_dataFile.c_str(), &current) == 0 &&
        indexed.st_dev == current.st_dev && indexed.st_ino == current.st_ino &&
        current.st_size >= 0 && static_cast<uint64_t>(current.st_size) == _validDataSize) {
        return true;
    }
    return scanLocked();
}

bool ReplayHistoryStore::scanLocked() const
{
    _index.clear();
    _persistedRecordCount = 0;
    _incompatibleRecordCount = 0;
    std::ifstream input(_dataFile, std::ios::binary);
    if (!input) {
        return false;
    }
    input.seekg(0, std::ios::end);
    const auto endPosition = input.tellg();
    if (endPosition < 0) {
        return false;
    }
    const auto fileSize = static_cast<uint64_t>(endPosition);
    input.seekg(0, std::ios::beg);
    uint64_t offset = 0;
    while (offset <= fileSize && fileSize - offset >= HEADER_SIZE) {
        uint32_t magic = 0;
        uint32_t version = 0;
        int64_t eventTimeUs = 0;
        uint64_t payloadSize = 0;
        uint32_t checksum = 0;
        if (!readValue(input, magic) || !readValue(input, version) ||
            !readValue(input, eventTimeUs) || !readValue(input, payloadSize) ||
            !readValue(input, checksum)) {
            return false;
        }
        if (magic != RECORD_MAGIC ||
            (version != LEGACY_RECORD_VERSION && version != RECORD_VERSION) ||
            eventTimeUs <= 0 || payloadSize > MAX_PAYLOAD_SIZE ||
            payloadSize > fileSize - offset - HEADER_SIZE) {
            break;
        }
        std::array<char, 64 * 1024> payloadBuffer;
        uint64_t remaining = payloadSize;
        uint32_t actualChecksum = 0;
        int32_t dataVersion = 0;
        bool hasDataVersion = false;
        while (remaining > 0) {
            const auto readSize = static_cast<std::streamsize>(
                std::min<uint64_t>(remaining, payloadBuffer.size()));
            if (!input.read(payloadBuffer.data(), readSize)) {
                return false;
            }
            actualChecksum = autil::CRC32C::Extend(
                actualChecksum, payloadBuffer.data(), readSize);
            if (!hasDataVersion && readSize >= sizeof(dataVersion)) {
                autil::DataBuffer versionBuffer(payloadBuffer.data(),
                                                sizeof(dataVersion));
                versionBuffer.read(dataVersion);
                hasDataVersion = true;
            }
            remaining -= static_cast<uint64_t>(readSize);
        }
        if (actualChecksum != checksum) {
            break;
        }
        ++_persistedRecordCount;
        const bool compatible = hasDataVersion &&
            ((version == LEGACY_RECORD_VERSION &&
              legacyDataVersionCompatible(dataVersion)) ||
             (version == RECORD_VERSION &&
              dataVersion == HISTORY_PAYLOAD_VERSION));
        if (compatible) {
            _index.push_back({eventTimeUs, offset, offset + HEADER_SIZE,
                              payloadSize, checksum, version});
        } else {
            ++_incompatibleRecordCount;
        }
        offset += HEADER_SIZE + payloadSize;
        input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!input) {
            break;
        }
    }
    input.close();
    if (offset != fileSize) {
        std::error_code error;
        std::filesystem::resize_file(_dataFile, offset, error);
        if (error) {
            return false;
        }
    }
    _validDataSize = offset;
    std::sort(_index.begin(), _index.end(), [](const auto &left, const auto &right) {
        return left.eventTimeUs != right.eventTimeUs
                   ? left.eventTimeUs < right.eventTimeUs
                   : left.recordOffset < right.recordOffset;
    });
    retainNewestLocked();
    if (_indexedFd >= 0) {
        ::close(_indexedFd);
    }
    _indexedFd = ::open(_dataFile.c_str(), O_RDONLY | O_CLOEXEC);
    return _indexedFd >= 0;
}

bool ReplayHistoryStore::append(const NaviSchedScheduleRequest &request,
                                int64_t eventTimeUs)
{
    if (!request.valid() || eventTimeUs <= 0) {
        return false;
    }
    autil::DataBuffer buffer;
    serializeHistoryRequest(buffer, request);
    const auto payloadSize = static_cast<uint64_t>(buffer.getDataLen());
    if (payloadSize > MAX_PAYLOAD_SIZE) {
        return false;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    if (_dataFile.empty()) {
        _volatileSamples.push_back({request, eventTimeUs});
        std::sort(_volatileSamples.begin(), _volatileSamples.end(),
                  [](const auto &left, const auto &right) {
                      return left.eventTimeUs < right.eventTimeUs;
                  });
        if (_volatileSamples.size() > _historyLimit) {
            _volatileSamples.erase(_volatileSamples.begin());
        }
        return true;
    }
    HistoryDirectoryLock diskLock(std::filesystem::path(_dataFile).parent_path());
    if (!diskLock.valid() || !refreshLocked()) {
        return false;
    }
    if (_persistedRecordCount >= maxPersistedRecordCount(_historyLimit) &&
        !compactLocked()) {
        return false;
    }
    std::error_code fileError;
    const auto fileSize = std::filesystem::file_size(_dataFile, fileError);
    if (fileError) {
        return false;
    }
    if (fileSize != _validDataSize) {
        std::filesystem::resize_file(_dataFile, _validDataSize, fileError);
        if (fileError) {
            return false;
        }
    }
    const auto checksum = autil::CRC32C::Value(buffer.getData(), buffer.getDataLen());
    std::ofstream output(_dataFile, std::ios::binary | std::ios::app);
    if (!output || !writeValue(output, RECORD_MAGIC) ||
        !writeValue(output, RECORD_VERSION) || !writeValue(output, eventTimeUs) ||
        !writeValue(output, payloadSize) || !writeValue(output, checksum) ||
        !output.write(buffer.getData(), buffer.getDataLen())) {
        return false;
    }
    output.flush();
    if (!output) {
        return false;
    }
    output.close();
    if (!output) {
        return false;
    }
    const auto recordOffset = _validDataSize;
    _validDataSize += HEADER_SIZE + payloadSize;
    ++_persistedRecordCount;
    _index.push_back({eventTimeUs, recordOffset, recordOffset + HEADER_SIZE,
                      payloadSize, checksum, RECORD_VERSION});
    std::sort(_index.begin(), _index.end(), [](const auto &left, const auto &right) {
        return left.eventTimeUs != right.eventTimeUs
                   ? left.eventTimeUs < right.eventTimeUs
                   : left.recordOffset < right.recordOffset;
    });
    retainNewestLocked();
    if (_persistedRecordCount > _historyLimit * 2) {
        compactLocked();
    }
    return true;
}

void ReplayHistoryStore::retainNewestLocked() const
{
    if (_index.size() > _historyLimit) {
        _index.erase(_index.begin(), _index.end() - _historyLimit);
    }
}

ReplayHistorySummary ReplayHistoryStore::getSummary(
    const std::string &clientId) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    ReplayHistorySummary summary;
    summary.clientId = clientId;
    if (_dataFile.empty()) {
        summary.sampleCount = _volatileSamples.size();
        if (!_volatileSamples.empty()) {
            summary.earliestEventTimeUs = _volatileSamples.front().eventTimeUs;
            summary.latestEventTimeUs = _volatileSamples.back().eventTimeUs;
        }
    } else {
        HistoryDirectoryLock diskLock(std::filesystem::path(_dataFile).parent_path());
        if (!diskLock.valid() || !refreshLocked()) {
            return summary;
        }
        summary.sampleCount = _index.size();
        if (!_index.empty()) {
            summary.earliestEventTimeUs = _index.front().eventTimeUs;
            summary.latestEventTimeUs = _index.back().eventTimeUs;
        }
    }
    summary.replayable = summary.sampleCount > 0;
    return summary;
}

bool ReplayHistoryStore::loadTimeRange(
    int64_t startTimeUs, int64_t endTimeUs,
    std::vector<ReplayHistorySample> &samples, std::string &error) const
{
    samples.clear();
    error.clear();
    if (startTimeUs <= 0 || endTimeUs <= startTimeUs) {
        error = "invalid replay history time range";
        return false;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    if (_dataFile.empty()) {
        for (const auto &sample : _volatileSamples) {
            if (sample.eventTimeUs >= startTimeUs &&
                sample.eventTimeUs <= endTimeUs) {
                samples.push_back(sample);
            }
        }
        return true;
    }
    HistoryDirectoryLock diskLock(std::filesystem::path(_dataFile).parent_path());
    if (!diskLock.valid() || !refreshLocked()) {
        error = "lock or refresh replay history failed";
        return false;
    }
    std::vector<IndexEntry> selected;
    for (const auto &entry : _index) {
        if (entry.eventTimeUs >= startTimeUs && entry.eventTimeUs <= endTimeUs) {
            selected.push_back(entry);
        }
    }
    return loadLocked(selected, samples, error);
}

bool ReplayHistoryStore::loadRecent(
    size_t count, std::vector<ReplayHistorySample> &samples,
    std::string &error) const
{
    samples.clear();
    error.clear();
    if (count == 0) {
        error = "recent query count must be positive";
        return false;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    if (_dataFile.empty()) {
        if (count > _volatileSamples.size()) {
            return true;
        }
        samples.assign(_volatileSamples.end() - count, _volatileSamples.end());
        return true;
    }
    HistoryDirectoryLock diskLock(std::filesystem::path(_dataFile).parent_path());
    if (!diskLock.valid() || !refreshLocked()) {
        error = "lock or refresh replay history failed";
        return false;
    }
    if (count > _index.size()) {
        return true;
    }
    std::vector<IndexEntry> selected(_index.end() - count, _index.end());
    return loadLocked(selected, samples, error);
}

bool ReplayHistoryStore::loadLocked(
    const std::vector<IndexEntry> &entries,
    std::vector<ReplayHistorySample> &samples, std::string &error) const
{
    if (entries.empty()) {
        return true;
    }
    std::ifstream input(_dataFile, std::ios::binary);
    if (!input) {
        error = "open replay history data failed";
        return false;
    }
    samples.reserve(entries.size());
    for (const auto &entry : entries) {
        std::string payload(entry.payloadSize, '\0');
        input.seekg(static_cast<std::streamoff>(entry.payloadOffset), std::ios::beg);
        if (!input || !input.read(payload.data(), payload.size())) {
            error = "read replay history record failed";
            samples.clear();
            return false;
        }
        if (autil::CRC32C::Value(payload.data(), payload.size()) != entry.checksum) {
            error = "replay history record checksum mismatch";
            samples.clear();
            return false;
        }
        try {
            autil::DataBuffer buffer(payload.data(), payload.size());
            ReplayHistorySample sample;
            sample.eventTimeUs = entry.eventTimeUs;
            const bool valid = entry.recordVersion == LEGACY_RECORD_VERSION
                                   ? deserializeLegacyHistoryRequest(
                                         buffer, sample.request)
                                   : entry.recordVersion == RECORD_VERSION &&
                                         deserializeHistoryRequest(
                                             buffer, sample.request);
            if (!valid) {
                error = "invalid replay history request";
                samples.clear();
                return false;
            }
            samples.push_back(std::move(sample));
        } catch (...) {
            error = "deserialize replay history request failed";
            samples.clear();
            return false;
        }
    }
    return true;
}

bool ReplayHistoryStore::compactLocked()
{
    if (_dataFile.empty()) {
        return true;
    }
    const auto temporary = _dataFile + COMPACTION_SUFFIX;
    std::ifstream input(_dataFile, std::ios::binary);
    if (!input) {
        return false;
    }
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    std::vector<IndexEntry> compacted;
    compacted.reserve(_index.size());
    uint64_t offset = 0;
    for (const auto &entry : _index) {
        const auto recordSize = HEADER_SIZE + entry.payloadSize;
        std::string record(recordSize, '\0');
        input.seekg(static_cast<std::streamoff>(entry.recordOffset), std::ios::beg);
        if (!input || !input.read(record.data(), record.size()) ||
            !output.write(record.data(), record.size())) {
            output.close();
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return false;
        }
        compacted.push_back({entry.eventTimeUs, offset, offset + HEADER_SIZE,
                             entry.payloadSize, entry.checksum,
                             entry.recordVersion});
        offset += recordSize;
    }
    output.flush();
    if (!output) {
        output.close();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    output.close();
    if (!output) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    input.close();
    std::error_code error;
    std::filesystem::rename(temporary, _dataFile, error);
    if (error) {
        error.clear();
        std::filesystem::remove(temporary, error);
        return false;
    }
    _index = std::move(compacted);
    _validDataSize = offset;
    _persistedRecordCount = _index.size();
    _incompatibleRecordCount = 0;
    return true;
}

bool ReplayHistoryStore::persistent() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return !_dataFile.empty();
}

size_t ReplayHistoryStore::residentRequestCount() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _volatileSamples.size();
}

} // namespace navi_sched
