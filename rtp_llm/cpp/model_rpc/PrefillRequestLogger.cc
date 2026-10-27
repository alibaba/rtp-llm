#include "rtp_llm/cpp/model_rpc/PrefillRequestLogger.h"

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>
#include "autil/TimeUtility.h"
#include "google/protobuf/util/json_util.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
constexpr size_t kQueueMaxBytes = 200ULL * 1024 * 1024;

int openPrivateDirectory(const std::string& log_path) {
    std::filesystem::create_directories(log_path);
    const auto path = std::filesystem::path(log_path) / "prefill_requests";
    if (::mkdir(path.c_str(), 0700) != 0 && errno != EEXIST) {
        throw std::system_error(errno, std::generic_category(), "create prefill request directory");
    }
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        throw std::system_error(errno, std::generic_category(), "open prefill request directory");
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_uid != ::geteuid() || ::fchmod(fd, 0700) != 0) {
        ::close(fd);
        throw std::runtime_error("prefill request directory must be owned and accessible only by the current user");
    }
    return fd;
}
}  // namespace

class PrefillRequestLogger::Worker {
public:
    Worker(const std::string& log_path, int64_t rank_id, int backup_count, size_t max_file_bytes):
        filename_("prefill_request_r" + std::to_string(rank_id) + "_p" + std::to_string(::getpid()) + ".log"),
        backup_count_(std::max(backup_count, 1)),
        max_file_bytes_(std::max<size_t>(max_file_bytes, 1)) {
        dir_fd_ = openPrivateDirectory(log_path);
        try {
            openFile();
            thread_ = std::thread([this]() { run(); });
        } catch (...) {
            if (fd_ >= 0) {
                ::close(fd_);
            }
            ::close(dir_fd_);
            throw;
        }
    }
    ~Worker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_one();
        thread_.join();
        if (fd_ >= 0) {
            ::close(fd_);
        }
        ::close(dir_fd_);
    }
    void enqueue(std::string record) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (disabled_ || stopping_ || record.size() > kQueueMaxBytes - pending_bytes_) {
            if (++dropped_ == 1 || dropped_ % 1000 == 0) {
                RTP_LLM_LOG_WARNING("Dropped prefill request log: writer unavailable or queue full; dropped=%zu",
                                    dropped_);
            }
            return;
        }
        queue_.push_back(std::move(record));
        pending_bytes_ += queue_.back().size();
        ready_.notify_one();
    }

private:
    void openFile() {
        fd_ = ::openat(dir_fd_, filename_.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd_ < 0) {
            throw std::system_error(errno, std::generic_category(), "open prefill request log");
        }
        struct stat st{};
        if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != ::geteuid() || st.st_nlink != 1
            || ::fchmod(fd_, 0600) != 0) {
            throw std::runtime_error("prefill request log must be a private regular file owned by the current user");
        }
        file_bytes_ = st.st_size;
    }
    void rotate() {
        ::close(fd_);
        fd_ = -1;
        for (int i = backup_count_; i > 0; --i) {
            const auto from = i == 1 ? filename_ : filename_ + "." + std::to_string(i - 1);
            const auto to   = filename_ + "." + std::to_string(i);
            if (::renameat(dir_fd_, from.c_str(), dir_fd_, to.c_str()) != 0 && errno != ENOENT) {
                throw std::system_error(errno, std::generic_category(), "rotate prefill request log");
            }
        }
        openFile();
    }
    void write(const std::string& record) {
        if (file_bytes_ > 0 && (file_bytes_ >= max_file_bytes_ || record.size() > max_file_bytes_ - file_bytes_)) {
            rotate();
        }
        size_t offset = 0;
        while (offset < record.size()) {
            const auto written = ::write(fd_, record.data() + offset, record.size() - offset);
            if (written < 0 && errno == EINTR) {
                continue;
            }
            if (written <= 0) {
                throw std::system_error(
                    written < 0 ? errno : EIO, std::generic_category(), "write prefill request log");
            }
            offset += written;
        }
        file_bytes_ += record.size();
    }
    void run() {
        while (true) {
            std::string record;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this]() { return stopping_ || !queue_.empty(); });
                if (queue_.empty()) {
                    return;
                }
                record = std::move(queue_.front());
                queue_.pop_front();
            }
            try {
                write(record);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("Disabled prefill request log after write failure: %s", e.what());
                std::lock_guard<std::mutex> lock(mutex_);
                disabled_ = true;
                queue_.clear();
                pending_bytes_ = 0;
                return;
            }
            std::lock_guard<std::mutex> lock(mutex_);
            pending_bytes_ -= record.size();
        }
    }

    std::string             filename_;
    int                     backup_count_;
    size_t                  max_file_bytes_;
    int                     dir_fd_        = -1;
    int                     fd_            = -1;
    size_t                  file_bytes_    = 0;
    size_t                  pending_bytes_ = 0;
    size_t                  dropped_       = 0;
    bool                    stopping_      = false;
    bool                    disabled_      = false;
    std::mutex              mutex_;
    std::condition_variable ready_;
    std::deque<std::string> queue_;
    std::thread             thread_;
};

PrefillRequestLogger::PrefillRequestLogger(const std::string& log_path,
                                           int64_t            rank_id,
                                           int                backup_count,
                                           size_t             max_file_bytes):
    worker_(std::make_unique<Worker>(log_path, rank_id, backup_count, max_file_bytes)), rank_id_(rank_id) {}

PrefillRequestLogger::~PrefillRequestLogger() = default;

void PrefillRequestLogger::log(const GenerateInputPB& request) const {
    try {
        google::protobuf::util::JsonPrintOptions options;
        options.preserve_proto_field_names = true;
        std::string request_json;
        const auto  status = google::protobuf::util::MessageToJsonString(request, &request_json, options);
        if (!status.ok()) {
            RTP_LLM_LOG_WARNING(
                "Failed to serialize prefill request [%ld]: %s", request.request_id(), status.ToString().c_str());
            return;
        }
        auto record =
            "{\"schema_version\":1,\"record_type\":\"prefill_request\",\"rank_id\":" + std::to_string(rank_id_)
            + ",\"arrival_time_us\":" + std::to_string(autil::TimeUtility::currentTimeInMicroSeconds())
            + ",\"request\":" + request_json + "}\n";
        worker_->enqueue(std::move(record));
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("Failed to log prefill request [%ld]: %s", request.request_id(), e.what());
    }
}

}  // namespace rtp_llm
