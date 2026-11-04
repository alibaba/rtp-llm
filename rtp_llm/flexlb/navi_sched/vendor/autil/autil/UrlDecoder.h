#pragma once
#include <string>

namespace autil {
class UrlDecoder {
private:
    UrlDecoder();
    ~UrlDecoder();
    UrlDecoder(const UrlDecoder &);
    UrlDecoder &operator=(const UrlDecoder &);

public:
    static std::string Decode(const std::string &str);
    static std::string Decode(const char *str, size_t size);
    static char *Decode(const char *s);

private:
    static char DecodeChar(const char *bytes);
};

} // namespace autil
