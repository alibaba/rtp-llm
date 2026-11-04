#include "alog/LogStream.h"

#include "LoggingEvent.h"

namespace alog {
LogStreamBuf::LogStreamBuf() {
    const int len = Logger::MAX_MESSAGE_LENGTH;
    _buf = new char[len];
    setp(_buf, _buf + len - 2);
}

LogStreamBuf::~LogStreamBuf() { delete[] _buf; }

LogStream::LogStream(Logger *logger, uint32_t level, const char *file, const int line, const char *func, int64_t ctr)
    : _logger(logger), _level(level), _file(file), _line(line), _function(func), _ctr(ctr), _self(this) {
    rdbuf(&_streambuf);
}

LogStream::~LogStream() {
    std::string msg(_streambuf.dataStart(), _streambuf.dataLength());
    if (nullptr != _logger) {
        LoggingEvent event(
            _logger->m_loggerName, std::move(msg), _level, std::string(_file), _line, std::string(_function));
        _logger->_log(event);
    }
}

std::ostream &operator<<(std::ostream &os, const alog::PRIVATE_Counter &) {
    alog::LogStream *log = dynamic_cast<alog::LogStream *>(&os);
    if (log && log == log->self()) {
        os << log->getCount();
    } else {
        os << "You must use COUNTER with LogStream";
    }
    return os;
}

CheckOpMessageBuilder::CheckOpMessageBuilder(const char *exprtext) : _stream(new std::ostringstream) {
    *_stream << exprtext << " (";
}

CheckOpMessageBuilder::~CheckOpMessageBuilder() { delete _stream; }

std::ostream *CheckOpMessageBuilder::ForVar2() {
    *_stream << " vs ";
    return _stream;
}

std::string *CheckOpMessageBuilder::NewString() {
    *_stream << ")";
    return new std::string(_stream->str());
}

template <>
void MakeCheckOpValueString(std::ostream *os, const char &v) {
    if (v >= 32 && v <= 126) {
        (*os) << "'" << v << "'";
    } else {
        (*os) << "char value " << static_cast<short>(v);
    }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const signed char &v) {
    if (v >= 32 && v <= 126) {
        (*os) << "'" << v << "'";
    } else {
        (*os) << "signed char value " << static_cast<short>(v);
    }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const unsigned char &v) {
    if (v >= 32 && v <= 126) {
        (*os) << "'" << v << "'";
    } else {
        (*os) << "unsigned char value " << static_cast<unsigned short>(v);
    }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const std::nullptr_t &v) {
    (*os) << "nullptr";
}

#define DEFINE_CHECK_STROP_IMPL(name, func, expected)                                                                  \
    std::string *Check##func##expected##Impl(const char *s1, const char *s2, const char *names) {                      \
        bool equal = s1 == s2 || (s1 && s2 && !func(s1, s2));                                                          \
        if (equal == expected) {                                                                                       \
            return NULL;                                                                                               \
        } else {                                                                                                       \
            std::ostringstream ss;                                                                                     \
            if (!s1) {                                                                                                 \
                s1 = "";                                                                                               \
            }                                                                                                          \
            if (!s2) {                                                                                                 \
                s2 = "";                                                                                               \
            }                                                                                                          \
            ss << #name " failed: " << names << " (" << s1 << " vs " << s2 << ")";                                     \
            return new std::string(ss.str());                                                                          \
        }                                                                                                              \
    }
DEFINE_CHECK_STROP_IMPL(CHECK_STREQ, strcmp, true)
DEFINE_CHECK_STROP_IMPL(CHECK_STRNE, strcmp, false)
DEFINE_CHECK_STROP_IMPL(CHECK_STRCASEEQ, strcasecmp, true)
DEFINE_CHECK_STROP_IMPL(CHECK_STRCASENE, strcasecmp, false)
#undef DEFINE_CHECK_STROP_IMPL

} // namespace alog
