#pragma once

#include <string>

namespace autil {
class UrlEncode {
public:
    static int encode(const char *str, const int strSize, char *result, const int resultSize);
    static int decode(const char *str, const int strSize, char *result, const int resultSize);
    static std::string decode(const std::string &s);
    static std::string encode(const std::string &s);
};

} // namespace autil
