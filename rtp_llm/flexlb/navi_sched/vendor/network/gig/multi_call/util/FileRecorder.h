#pragma once

#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
namespace multi_call {

class FileRecorder
{
public:
    FileRecorder();
    ~FileRecorder();

private:
    FileRecorder(const FileRecorder &);
    FileRecorder &operator=(const FileRecorder &);

public:
    static void recordSnapshot(const std::string &content, size_t logCount,
                               const std::string &prefix = "snapshot",
                               const std::string &realpath = "");
    static void newRecord(const std::string &content, size_t logCount, const std::string &dir,
                          const std::string &suffix, const std::string &realpath = "");
    static bool hasSuffix(const std::string &str, const std::string &suffix);

private:
    static std::string join(const std::string &path, const std::string &file);
    static bool listDir(const std::string &dirName, std::vector<std::string> &fileList);
    static bool removeFile(const std::string &path);
    static bool writeFile(const std::string &srcFileName, const std::string &content);
    static bool getCurrentPath(std::string &path);
    static bool checkAndCreateDir(const std::string &dir);
    static std::string getFileName(const std::string &suffix);

private:
    AUTIL_LOG_DECLARE();
};

} // namespace multi_call

