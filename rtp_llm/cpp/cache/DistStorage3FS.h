#pragma once

#include "rtp_llm/cpp/cache/DistStorage.h"
#include "rtp_llm/cpp/cache/DistStorage3FSFile.h"
#include "rtp_llm/cpp/cache/ThreeFSCudaUtil.h"

namespace rtp_llm::threefs {

class DistStorage3FS: public DistStorage {
public:
    DistStorage3FS(const kmonitor::MetricsReporterPtr& metrics_reporter);
    virtual ~DistStorage3FS();

public:
    bool init(const DistStorage3FSInitParams& init_params);

    bool lookup(const DistStorage::Item& item) override;
    bool get(const DistStorage::Item& item) override;
    bool put(const DistStorage::Item& item) override;
    bool del(const DistStorage::Item& item) override;

private:
    bool checkInitParams(const DistStorage3FSInitParams& init_params) const;

    std::shared_ptr<threefs::DistStorage3FSFile> getFile(const DistStorage::Item& item, bool read = true);
    void                                         removeFile(const DistStorage::Item& item);
    std::string                                  makeFilepath(const std::map<std::string, std::string>& metas) const;

    bool initIovHandle(threefs::ThreeFSIovHandle&                       handle,
                       size_t                                           iov_block_size,
                       size_t                                           iov_size,
                       const std::shared_ptr<threefs::ThreeFSCudaUtil>& cuda_util);
    void releaseIovHandle(threefs::ThreeFSIovHandle& handle);
    void deleteIovShm() const;

    struct hf3fs_iov* createIov(const std::string& mountpoint, size_t iov_size, size_t iov_block_size) const;
    void              releaseIov(struct hf3fs_iov* iov) const;

    void                               reportMetrics() const;
    std::tuple<size_t, size_t, size_t> getFileSystemInfo() const;

private:
    kmonitor::MetricsReporterPtr metrics_reporter_;
    DistStorage3FSInitParams     init_params_;

    // for read & write
    threefs::ThreeFSIovHandle read_iov_handle_;
    threefs::ThreeFSIovHandle write_iov_handle_;

    // for 3fs async write
    std::shared_ptr<autil::LockFreeThreadPool> write_thread_pool_;

    // TODO change to LRUMap, avoid exceed max handler num
    const uint32_t                                                                kMaxFileNum = 1000;
    std::unordered_map<std::string, std::shared_ptr<threefs::DistStorage3FSFile>> file_map_;
    std::shared_mutex                                                             file_map_mutex_;

    // for metric
    std::atomic<bool> stop_report_metrics_{false};
    std::thread       report_metrics_thread_;
};

}  // namespace rtp_llm::threefs