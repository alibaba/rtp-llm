#include "autil/URLUtil.h"

#include <assert.h>
#include <ctype.h>
#include <iosfwd>
#include <stdlib.h>
#include <strings.h>

using namespace std;

namespace autil {

URLUtil::URLUtil() {}

URLUtil::~URLUtil() {}

string URLUtil::decode(const string &str) {
    size_t nlen = str.length();
    char *dstBuf = (char *)malloc(nlen + 1);
    assert(dstBuf);
    bzero(dstBuf, nlen + 1);
    char *dst = dstBuf;
    const char *src = str.c_str();
    char temp[3]; // store two char for strtol, end with '\0'
    temp[2] = '\0';

    for (size_t i = 0; i <= nlen;) {
        if (src[i] == '+') {
            *dst++ = ' ';
            ++i;
        } else if (src[i] == '%' && i + 2 <= nlen && isxdigit(src[i + 1]) && isxdigit(src[i + 2])) {
            // convert hex string to integer
            temp[0] = src[i + 1];
            temp[1] = src[i + 2];
            *dst++ = strtol(temp, NULL, 16);
            i += 3;
        } else {
            *dst++ = src[i];
            ++i;
        }
    }
    string ss(dstBuf);
    free(dstBuf);
    return ss;
}

} // namespace autil
