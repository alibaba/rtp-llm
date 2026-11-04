#include "autil/UrlDecoder.h"

#include <cstddef>
#include <ctype.h>

#include "string.h"

using namespace std;

namespace autil {

UrlDecoder::UrlDecoder() {}

UrlDecoder::~UrlDecoder() {}

string UrlDecoder::Decode(const string &str) { return Decode(str.data(), str.size()); }

string UrlDecoder::Decode(const char *str, size_t size) {
    string result;
    result.reserve(size);
    const char *data = str;
    for (size_t len = size; len > 0; --len) {
        if (*data == '+') {
            result.append(" ");
        } else if (*data == '%' && len >= 2 && isxdigit(*(data + 1)) && isxdigit(*(data + 2))) {
            result.append(1, (char)DecodeChar(data + 1));
            data += 2;
            len -= 2;
        } else {
            result.append(1, *data);
        }
        ++data;
    }
    return result;
}

char *UrlDecoder::Decode(const char *str) {
    char *res = new char[strlen(str) + 1];
    char *cur = res;

    for (; *str != '\0'; ++str, ++cur) {
        if (*str == '+') {
            *cur = ' ';
        } else if (*str == '%' && *(str + 1) != '\0' && *(str + 2) != '\0') {
            *cur = DecodeChar(str + 1);
            str += 2;
        } else {
            *cur = *str;
        }
    }
    *cur = '\0';
    return res;
}

char UrlDecoder::DecodeChar(const char *bytes) {
    int ch = tolower(*bytes++);
    int value = (ch >= '0' && ch <= '9' ? ch - '0' : ch - 'a' + 10) << 4;
    ch = tolower(*bytes);
    value += (ch >= '0' && ch <= '9') ? ch - '0' : ch - 'a' + 10;
    return (char)value;
}

} // namespace autil
