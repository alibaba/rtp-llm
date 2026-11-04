#pragma once

#include <stddef.h>
#include <string>

namespace autil {

class FileRecorder {
public:
    static void newRecord(const std::string &content,
                          const std::string &dir,
                          const std::string &suffix,
                          size_t id,
                          size_t keepCount = _maxCount);
    static bool hasSuffix(const std::string &str, const std::string &suffix);
    static const size_t _maxCount;
};

} // namespace autil
