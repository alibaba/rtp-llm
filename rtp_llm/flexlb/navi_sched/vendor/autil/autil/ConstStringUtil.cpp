#include "autil/ConstStringUtil.h"

#include <iosfwd>
#include <stdio.h>

using namespace std;
namespace autil {

std::vector<autil::StringView>
ConstStringUtil::split(const autil::StringView &text, const std::string &sepStr, bool ignoreEmpty) {
    std::vector<autil::StringView> vec;
    split(vec, text, sepStr, ignoreEmpty);
    return vec;
}

std::vector<autil::StringView>
ConstStringUtil::split(const autil::StringView &text, const char &sepChar, bool ignoreEmpty) {
    std::vector<autil::StringView> vec;
    split(vec, text, sepChar, ignoreEmpty);
    return vec;
}

void ConstStringUtil::split(std::vector<autil::StringView> &vec,
                            const autil::StringView &text,
                            const char &sepChar,
                            bool ignoreEmpty) {
    split(vec, text, std::string(1, sepChar), ignoreEmpty);
}

void ConstStringUtil::split(std::vector<autil::StringView> &vec,
                            const autil::StringView &text,
                            const std::string &sepStr,
                            bool ignoreEmpty) {
    size_t n = 0, old = 0;
    while (n != std::string::npos) {
        n = text.find(sepStr, n);
        if (n != std::string::npos) {
            if (!ignoreEmpty || n != old)
                vec.push_back(text.substr(old, n - old));
            n += sepStr.length();
            old = n;
        }
    }

    if (!ignoreEmpty || old < text.length()) {
        vec.push_back(text.substr(old, text.length() - old));
    }
}

StringView
ConstStringUtil::join(const std::vector<StringView> &array, const StringView &seperator, mem_pool::Pool *pool) {
    size_t targetLength = 0;
    if (!pool || array.empty()) {
        return StringView();
    } else {
        targetLength += (array.size() - 1) * seperator.size();
    }
    for (auto &val : array) {
        targetLength += val.size();
    }
    char *buffer = (char *)pool->allocate(targetLength);
    char *cursor = buffer;
    for (size_t i = 0; i < array.size(); i++) {
        auto &val = array[i];
        if (!val.empty()) {
            memcpy(cursor, val.data(), val.size());
            cursor += val.size();
        }
        if ((i + 1) != array.size()) {
            memcpy(cursor, seperator.data(), seperator.size());
            cursor += seperator.size();
        }
    }
    assert((size_t)(cursor - buffer) == targetLength);
    return StringView(buffer, targetLength);
}

} // namespace autil
