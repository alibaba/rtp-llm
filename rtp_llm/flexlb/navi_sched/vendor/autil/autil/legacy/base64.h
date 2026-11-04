#pragma once

#include <iostream>
#include <stddef.h>
#include <string>

#include "autil/legacy/exception.h"

namespace autil {
namespace legacy {

class BadBase64Exception : public ExceptionBase {
public:
    AUTIL_LEGACY_DEFINE_EXCEPTION(BadBase64Exception, ExceptionBase);
};

void Base64Encoding(std::istream &,
                    std::ostream &,
                    char makeupChar = '=',
                    const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/");
void Base64Decoding(std::istream &, std::ostream &, char plus = '+', char slash = '/');
void Base64DecodingEx(std::istream &, std::ostream &, char makeupChar = '=', char plus = '+', char slash = '/');

std::string Base64DecodeFast(const std::string &str);
std::string Base64EncodeFast(const std::string &str);

std::string Base64DecodeFast(const char *data, size_t size);
std::string Base64EncodeFast(const char *data, size_t size);

} // namespace legacy
} // namespace autil
