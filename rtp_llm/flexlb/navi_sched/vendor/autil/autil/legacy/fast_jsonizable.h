#pragma once

#ifndef RAPIDJSON_HAS_STDSTRING
#define RAPIDJSON_HAS_STDSTRING 1
#endif

#if defined(__SSE4_2__)
#ifndef AUTIL_RAPIDJSON_SSE42
#define AUTIL_RAPIDJSON_SSE42
#endif
#elif defined(__SSE2__)
#ifndef AUTIL_RAPIDJSON_SSE2
#define AUTIL_RAPIDJSON_SSE2
#endif
#endif

#include <deque>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "autil/LongHashValue.h"
#include "autil/legacy/any.h"
#include "autil/legacy/exception.h"
#include "autil/legacy/jsonizable_exception.h"
#include "autil/result/Errors.h"
#include "autil/result/Result.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace autil {
namespace legacy {

class FastJsonizable;
class Jsonizable;

typedef rapidjson::GenericValue<rapidjson::UTF8<>> RapidValue;
typedef rapidjson::GenericDocument<rapidjson::UTF8<>> RapidDocument;
typedef rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>, rapidjson::ASCII<>> RapidWriter;

template <bool b, class T = void>
using enable_if_t = typename std::enable_if<b, T>::type;

inline std::string FastToJsonString(const RapidValue &value);

// serializeToWriter declare begin
struct SerializeCtx {
    RapidWriter *writer = nullptr;
    bool debugMode = false;
    std::set<std::string> keySet;
    static thread_local bool CHECK_DUP_KEY; // CHECK_DUP_KEY = false by default

    SerializeCtx() = default;

    SerializeCtx(RapidWriter *writer) : SerializeCtx(writer, false) {}

    SerializeCtx(RapidWriter *writer, bool debugMode) : writer(writer), debugMode(debugMode) {}

    bool addKeyIfNotExist(const std::string &key) {
        if (!CHECK_DUP_KEY) {
            return true;
        }
        return keySet.insert(key).second;
    }

    RapidWriter *operator->() { return writer; }
};

class FastToJsonDupKeyPretectGuard {
public:
    FastToJsonDupKeyPretectGuard() {
        _oldCheckStatus = SerializeCtx::CHECK_DUP_KEY;
        SerializeCtx::CHECK_DUP_KEY = true;
    }

    ~FastToJsonDupKeyPretectGuard() { SerializeCtx::CHECK_DUP_KEY = _oldCheckStatus; }

private:
    bool _oldCheckStatus = false;
};

inline void serializeToWriter(SerializeCtx writer, const std::string &t);
inline void serializeToWriter(SerializeCtx writer, const char *t);
inline void serializeToWriter(SerializeCtx writer, const FastJsonizable &t);
inline void serializeToWriter(SerializeCtx writer, bool t);
template <typename T>
inline enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, void> serializeToWriter(SerializeCtx writer,
                                                                                                   const T &t);
template <typename T>
inline enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, void>
serializeToWriter(SerializeCtx writer, const T &t);
template <typename T>
inline enable_if_t<std::is_enum<T>::value, void> serializeToWriter(SerializeCtx writer, const T &t);
template <typename T>
inline enable_if_t<std::is_floating_point<T>::value, void> serializeToWriter(SerializeCtx writer, const T &t);
template <typename T, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::vector<T, A> &t);
template <typename T, typename C, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::set<T, C, A> &t);
template <typename T, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::deque<T, A> &t);
template <typename T, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::list<T, A> &t);
template <typename T, typename C, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::map<std::string, T, C, A> &t);
template <typename T, typename U, typename C, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::map<T, U, C, A> &t);
template <typename T, typename U, typename C, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::multimap<T, U, C, A> &t);
template <typename T, typename U>
inline void serializeToWriter(SerializeCtx writer, const std::pair<T, U> &t);
template <typename T>
inline void serializeToWriter(SerializeCtx writer, const T *t);
template <typename T>
inline void serializeToWriter(SerializeCtx writer, const std::shared_ptr<T> &t);
template <typename T>
inline void serializeToWriter(SerializeCtx writer, const std::shared_ptr<T> &t);
inline void serializeToWriter(SerializeCtx writer, const RapidValue *t);
inline void serializeToWriter(SerializeCtx writer, const ExceptionBase &t);
inline void serializeToWriter(SerializeCtx writer, const uint128_t &t);

// serializeToWriter declare end

// FromRapidValue declare begin
inline void FromRapidValue(std::string &t, const RapidValue &value);
inline void FromRapidValue(bool &t, const RapidValue &value);
template <typename T>
inline enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, void>
FromRapidValue(T &t, const RapidValue &value);
template <typename T>
inline enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, void>
FromRapidValue(T &t, const RapidValue &value);
template <typename T>
inline enable_if_t<std::is_enum<T>::value, void> FromRapidValue(T &t, const RapidValue &value);
template <typename T>
inline enable_if_t<std::is_floating_point<T>::value, void> FromRapidValue(T &t, const RapidValue &value);
template <typename T, typename A>
inline void FromRapidValue(std::vector<T, A> &t, const RapidValue &value);
template <typename T, typename C, typename A>
inline void FromRapidValue(std::set<T, C, A> &t, const RapidValue &value);
template <typename T, typename A>
inline void FromRapidValue(std::deque<T, A> &t, const RapidValue &value);
template <typename T, typename A>
inline void FromRapidValue(std::list<T, A> &t, const RapidValue &value);
template <typename T, typename C, typename A>
inline void FromRapidValue(std::map<std::string, T, C, A> &t, const RapidValue &value);
template <typename T, typename U, typename C, typename A>
inline void FromRapidValue(std::map<T, U, C, A> &t, const RapidValue &value);
template <typename T, typename U, typename C, typename A>
inline void FromRapidValue(std::multimap<T, U, C, A> &t, const RapidValue &value);
template <typename T, typename U>
inline void FromRapidValue(std::pair<T, U> &t, const RapidValue &value);
template <typename T>
inline void FromRapidValue(T *&t, const RapidValue &value);
template <typename T>
inline void FromRapidValue(std::shared_ptr<T> &t, const RapidValue &value);
template <typename T>
inline void FromRapidValue(std::shared_ptr<T> &t, const RapidValue &value);
inline void FromRapidValue(const RapidValue *&t, const RapidValue &value);
inline void FromRapidValue(ExceptionBase &t, const RapidValue &value);
inline void FromRapidValue(uint128_t &t, const RapidValue &value);
// FromRapidValue declare end

void FromRapidValue(Any &t, const RapidValue &value);
void serializeToWriter(SerializeCtx writer, const Any &a);

// serializeToWriter implemention begin
inline void serializeToWriter(SerializeCtx writer, const std::string &t) { writer->String(t); }

inline void serializeToWriter(SerializeCtx writer, const char *t) { writer->String(t); }

inline void serializeToWriter(SerializeCtx writer, bool t) { writer->Bool(t); }

template <typename T>
inline enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, void> serializeToWriter(SerializeCtx writer,
                                                                                                   const T &t) {
    writer->Int64((int64_t)t);
}

template <typename T>
inline enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, void>
serializeToWriter(SerializeCtx writer, const T &t) {
    writer->Uint64((uint64_t)t);
}

template <typename T>
inline enable_if_t<std::is_enum<T>::value, void> serializeToWriter(SerializeCtx writer, const T &t) {
    writer->Int((int32_t)t);
}

template <typename T>
inline enable_if_t<std::is_floating_point<T>::value, void> serializeToWriter(SerializeCtx writer, const T &t) {
    writer->Double((double)t);
}

#define SERIALIZE_AS_ARRAY(container, writer)                                                                          \
    {                                                                                                                  \
        writer->StartArray();                                                                                          \
        for (auto &v : container) {                                                                                    \
            serializeToWriter(writer, v);                                                                              \
        }                                                                                                              \
        writer->EndArray();                                                                                            \
    }

template <typename T, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::vector<T, A> &t) {
    SERIALIZE_AS_ARRAY(t, writer)
}

template <>
inline void serializeToWriter<bool, std::allocator<bool>>(SerializeCtx writer,
                                                          const std::vector<bool, std::allocator<bool>> &t) {
    writer->StartArray();
    for (bool v : t) {
        serializeToWriter(writer, v);
    }
    writer->EndArray();
}

template <typename T, typename C, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::set<T, C, A> &t) {
    SERIALIZE_AS_ARRAY(t, writer)
}

template <typename T, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::deque<T, A> &t) {
    SERIALIZE_AS_ARRAY(t, writer)
}

template <typename T, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::list<T, A> &t) {
    SERIALIZE_AS_ARRAY(t, writer)
}

template <typename T, typename C, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::map<std::string, T, C, A> &t) {
    writer->StartObject();
    for (auto &kv : t) {
        writer->Key(kv.first.c_str());
        serializeToWriter(writer, kv.second);
    }
    writer->EndObject();
}

template <typename T, typename U, typename C, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::map<T, U, C, A> &t) {
    writer->StartArray();
    for (auto &kv : t) {
        serializeToWriter(writer, kv);
    }
    writer->EndArray();
}

template <typename T, typename U, typename C, typename A>
inline void serializeToWriter(SerializeCtx writer, const std::multimap<T, U, C, A> &t) {
    writer->StartArray();
    for (auto &kv : t) {
        serializeToWriter(writer, kv);
    }
    writer->EndArray();
}

template <typename T, typename U>
inline void serializeToWriter(SerializeCtx writer, const std::pair<T, U> &t) {
    writer->StartArray();
    serializeToWriter(writer, t.first);
    serializeToWriter(writer, t.second);
    writer->EndArray();
}

template <typename T>
inline void serializeToWriter(SerializeCtx writer, const T *t) {
    if (t) {
        serializeToWriter(writer, *t);
    } else {
        writer->Null();
    }
}

template <typename T>
inline void serializeToWriter(SerializeCtx writer, const std::shared_ptr<T> &t) {
    serializeToWriter(writer, t.get());
}

inline void serializeToWriter(SerializeCtx writer, const RapidValue *t) { t->Accept(*writer.writer); }

inline void serializeToWriter(SerializeCtx writer, const ExceptionBase &t) {
    writer->StartObject();
    writer->Key("Message");
    writer->String(t.mMessage);
    writer->Key("File");
    writer->String(t.mFile);
    writer->Key("Function");
    writer->String(t.mFunction);
    writer->Key("Line");
    writer->Int(t.mLine);
    writer->Key("ClassName");
    writer->String(t.GetClassName());
    if (t.mNestedException) {
        writer->Key("Cause");
        serializeToWriter(writer, t.mNestedException);
    }
    writer->EndObject();
}

inline void serializeToWriter(SerializeCtx writer, const uint128_t &t) {
    writer->StartArray();
    writer->Int64(t.value[0]);
    writer->Int64(t.value[1]);
    writer->EndArray();
}

// serializeToWriter implemention end

// FromRapidValue implemention begin
inline void FromRapidValue(std::string &t, const RapidValue &value) {
    if (!value.IsString()) {
        AUTIL_LEGACY_THROW(TypeNotMatchException, "type not match, expect String but get: " + FastToJsonString(value));
    }
    t = value.GetString();
}

inline void FromRapidValue(bool &t, const RapidValue &value) {
    if (!value.IsBool()) {
        // todo: fill more info to exception
        AUTIL_LEGACY_THROW(TypeNotMatchException, "type not match, expect bool but get: " + FastToJsonString(value));
    }
    t = value.GetBool();
}

template <typename T>
inline enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, void>
FromRapidValue(T &t, const RapidValue &value) {
    if (!value.IsInt64()) {
        AUTIL_LEGACY_THROW(TypeNotMatchException, "type not match, expect int64 but get: " + FastToJsonString(value));
    }
    t = (T)value.GetInt64();
}

template <typename T>
inline enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, void>
FromRapidValue(T &t, const RapidValue &value) {
    if (!value.IsUint64()) {
        AUTIL_LEGACY_THROW(TypeNotMatchException, "type not match, expect uint64 but get: " + FastToJsonString(value));
    }
    t = (T)value.GetUint64();
}

template <typename T>
inline enable_if_t<std::is_enum<T>::value, void> FromRapidValue(T &t, const RapidValue &value) {
    if (!value.IsInt()) {
        AUTIL_LEGACY_THROW(TypeNotMatchException, "type not match, expect int but get: " + FastToJsonString(value));
    }
    t = (T)value.GetInt();
}

template <typename T>
inline enable_if_t<std::is_floating_point<T>::value, void> FromRapidValue(T &t, const RapidValue &value) {
    if (!value.IsNumber()) {
        AUTIL_LEGACY_THROW(TypeNotMatchException, "type not match, expect number but get: " + FastToJsonString(value));
    }
    t = (T)value.GetDouble();
}

#define DESERIALIZE_FROM_ARRAY(container, func, value)                                                                 \
    {                                                                                                                  \
        if (!value.IsArray()) {                                                                                        \
            AUTIL_LEGACY_THROW(TypeNotMatchException,                                                                  \
                               "type not match, expect Array but get: " + FastToJsonString(value));                    \
        }                                                                                                              \
        container.clear();                                                                                             \
        for (auto itr = value.Begin(); itr != value.End(); ++itr) {                                                    \
            T tmpValue;                                                                                                \
            FromRapidValue(tmpValue, *itr);                                                                            \
            container.func(tmpValue);                                                                                  \
        }                                                                                                              \
    }

template <typename T, typename A>
inline void FromRapidValue(std::vector<T, A> &t, const RapidValue &value) {
    DESERIALIZE_FROM_ARRAY(t, push_back, value)
}

template <typename T, typename C, typename A>
inline void FromRapidValue(std::set<T, C, A> &t, const RapidValue &value) {
    DESERIALIZE_FROM_ARRAY(t, insert, value)
}

template <typename T, typename A>
inline void FromRapidValue(std::deque<T, A> &t, const RapidValue &value) {
    DESERIALIZE_FROM_ARRAY(t, push_back, value)
}

template <typename T, typename A>
inline void FromRapidValue(std::list<T, A> &t, const RapidValue &value) {
    DESERIALIZE_FROM_ARRAY(t, push_back, value)
}

template <typename T, typename C, typename A>
inline void FromRapidValue(std::map<std::string, T, C, A> &t, const RapidValue &value) {
    if (!value.IsObject()) {
        AUTIL_LEGACY_THROW(TypeNotMatchException, "type not match, expect Object but get: " + FastToJsonString(value));
    }
    t.clear();
    for (auto itr = value.MemberBegin(); itr != value.MemberEnd(); ++itr) {
        std::string k = itr->name.GetString();
        T v;
        FromRapidValue(v, itr->value);
        t.insert(std::make_pair(k, v));
    }
}

template <typename T, typename U, typename C, typename A>
inline void FromRapidValue(std::map<T, U, C, A> &t, const RapidValue &value) {
    if (!value.IsArray()) {
        AUTIL_LEGACY_THROW(TypeNotMatchException, "type not match, expect Array but get: " + FastToJsonString(value));
    }
    t.clear();
    for (auto itr = value.Begin(); itr != value.End(); ++itr) {
        std::pair<T, U> tmpPair;
        FromRapidValue(tmpPair, *itr);
        t.insert(tmpPair);
    }
}

template <typename T, typename U, typename C, typename A>
inline void FromRapidValue(std::multimap<T, U, C, A> &t, const RapidValue &value) {
    if (!value.IsArray()) {
        AUTIL_LEGACY_THROW(TypeNotMatchException, "type not match, expect Array but get: " + FastToJsonString(value));
    }
    t.clear();
    for (auto itr = value.Begin(); itr != value.End(); ++itr) {
        std::pair<T, U> tmpPair;
        FromRapidValue(tmpPair, *itr);
        t.insert(tmpPair);
    }
}

template <typename T, typename U>
inline void FromRapidValue(std::pair<T, U> &t, const RapidValue &value) {
    if (!value.IsArray()) {
        AUTIL_LEGACY_THROW(TypeNotMatchException, "type not match, expect Array but get: " + FastToJsonString(value));
    }
    if (value.Size() != 2) {
        AUTIL_LEGACY_THROW(TypeNotMatchException,
                           "type not match, expect 2 element Array but get: " + FastToJsonString(value));
    }
    FromRapidValue(t.first, value[0]);
    FromRapidValue(t.second, value[1]);
}

template <typename T>
inline void FromRapidValue(T *&t, const RapidValue &value) {
    if (value.IsNull()) {
        t = NULL;
        return;
    }
#if __cplusplus >= 201103L
    std::unique_ptr<T> p(new T);
#else
    std::auto_ptr<T> p(new T);
#endif
    FromRapidValue(*p, value);
    t = p.release();
}

template <typename T>
inline void FromRapidValue(std::shared_ptr<T> &t, const RapidValue &value) {
    T *p = NULL;
    FromRapidValue(p, value);
    t.reset(p);
}

inline void FromRapidValue(const RapidValue *&t, const RapidValue &value) { t = &value; }

inline void FromRapidValue(ExceptionBase &t, const RapidValue &value) {
    if (!value.IsObject()) {
        AUTIL_LEGACY_THROW(TypeNotMatchException, "type not match, expect Object but get: " + FastToJsonString(value));
    }
    FromRapidValue(t.mMessage, value["Message"]);
    FromRapidValue(t.mLine, value["Line"]);
    if (value.HasMember("Cause")) {
        FromRapidValue(t.mNestedException, value["Cause"]);
    }
}

inline void FromRapidValue(uint128_t &t, const RapidValue &value) {
    if (!value.IsArray()) {
        AUTIL_LEGACY_THROW(TypeNotMatchException, "type not match, expect Array but get: " + FastToJsonString(value));
    }
    if (value.Size() != 2) {
        AUTIL_LEGACY_THROW(TypeNotMatchException,
                           "type not match, expect 2 element Array but get: " + FastToJsonString(value));
    }
    FromRapidValue(t.value[0], value[0]);
    FromRapidValue(t.value[1], value[1]);
}

// FromRapidValue implemention end
namespace {
template <typename T>
const char *FastToJsonStringImpl(const T &t, bool debugMode, rapidjson::StringBuffer &s) {
    RapidWriter writer(s);
    SerializeCtx ctx(&writer, debugMode);
    serializeToWriter(ctx, t);
    return s.GetString();
}

template <typename T>
std::string FastToJsonStringImpl(const T &t, bool debugMode) {
    rapidjson::StringBuffer s;
    const char *result = FastToJsonStringImpl(t, debugMode, s);
    return std::string(result);
}
} // namespace

template <typename T>
std::string FastToJsonString(const T &t) {
    return FastToJsonStringImpl(t, false);
}

template <typename T>
std::string FastToDebugJsonString(const T &t) {
    return FastToJsonStringImpl(t, true);
}

template <typename T>
const char *FastToJsonString(const T &t, rapidjson::StringBuffer &s) {
    return FastToJsonStringImpl(t, false, s);
}

template <typename T>
const char *FastToDebugJsonString(const T &t, rapidjson::StringBuffer &s) {
    return FastToJsonStringImpl(t, true, s);
}

inline std::string FastToJsonString(const RapidValue &value) {
    rapidjson::StringBuffer s;
    RapidWriter writer(s);
    value.Accept(writer);
    return s.GetString();
}

template <typename T>
std::string FastToJsonString(const T &t, bool isCompact, const std::string &prefix = "") {
    // FIXME
    // support isCompact && perfix
    return FastToJsonString(t);
}

template <typename T>
void FastFromJsonString(T &t, const std::string &str) {
    RapidDocument document;
    document.Parse(str.c_str());
    if (document.HasParseError()) {
        std::ostringstream errStream;
        errStream << "JSON parse error:" << document.GetParseError() << " offset:" << document.GetErrorOffset();
        AUTIL_LEGACY_THROW(WrongFormatJsonException, errStream.str());
    }
    FromRapidValue(t, document);
}

inline std::string RapidValue2Str(const RapidValue &value) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    value.Accept(writer);
    return buffer.GetString();
}

#ifdef DISABLE_LEGACY_JSON
template <typename T>
std::string ToJsonString(const T &t) {
    return FastToJsonString(t);
}

template <typename T>
std::tuple<bool, std::string> ToJsonStringNoException(const T &t, std::string &result) noexcept {
    try {
        result = ToJsonString(t);
    } catch (autil::legacy::ExceptionBase &e) {
        return std::make_tuple(false, e.GetExceptionHeader());
    }
    return std::make_tuple(true, "");
}

template <typename T>
[[deprecated]] std::string ToJsonString(const T &t, bool isCompact, const std::string &prefix = "") {
    return FastToJsonString(t, isCompact, prefix);
}

template <typename T>
[[deprecated]] std::tuple<bool, std::string>
ToJsonStringNoException(const T &t, bool isCompact, std::string &result, const std::string &prefix = "") noexcept {
    try {
        result = FastToJsonString(t, isCompact, prefix);
    } catch (autil::legacy::ExceptionBase &e) {
        return std::make_tuple(false, e.GetExceptionHeader());
    }
    return std::make_tuple(true, "");
}

template <typename T>
void FromJsonString(T &t, const std::string &str) {
    FastFromJsonString(t, str);
}

template <typename T>
[[nodiscard]] std::tuple<bool, std::string> FromJsonStringNoException(T &t, const std::string &str) noexcept {
    try {
        FromJsonString(t, str);
    } catch (autil::legacy::ExceptionBase &e) {
        return std::make_tuple(false, e.GetExceptionHeader());
    }
    return std::make_tuple(true, "");
}
#endif

template <typename T>
autil::Result<> FromRapidValueWithCheck(T &t, const RapidValue &value) {
    try {
        FromRapidValue(t, value);
    } catch (const std::exception &e) {
        return autil::result::RuntimeError::make(
            "from rapid value failed, err[%s] value[%s]", e.what(), FastToJsonString(value).c_str());
    }
    return {};
}

} // namespace legacy
} // namespace autil
