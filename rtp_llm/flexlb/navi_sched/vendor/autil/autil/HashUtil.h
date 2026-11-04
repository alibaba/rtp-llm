#pragma once

#include <functional>
#include <stdint.h>
#include <string.h>
#include <string>

#include "autil/HashAlgorithm.h"
#include "autil/LongHashValue.h"
#include "autil/MultiValueType.h"
#include "autil/StringView.h"
#include "autil/cityhash/city.h"

namespace autil {

template <class T>
class KeyHash {};

template <>
class KeyHash<char> {
public:
    inline uint64_t operator()(char key) const { return (uint64_t)key; }
};

template <>
class KeyHash<const char *> {
public:
    uint64_t operator()(const char *key) const { return HashAlgorithm::hashString64(key); }
};

template <>
class KeyHash<std::string> {
public:
    uint64_t operator()(const std::string &key) const { return HashAlgorithm::hashString64(key.c_str()); }
};

template <>
class KeyHash<long long int> {
public:
    uint64_t operator()(long long int key) const { return (uint64_t)key; }
};

template <>
class KeyHash<unsigned long long int> {
public:
    uint64_t operator()(unsigned long long int key) const { return (uint64_t)key; }
};

template <>
class KeyHash<uint64_t> {
public:
    uint64_t operator()(uint64_t key) const { return key; }
};

template <>
class KeyHash<int64_t> {
public:
    uint64_t operator()(int64_t key) const { return (uint64_t)key; }
};

template <>
class KeyHash<uint32_t> {
public:
    uint64_t operator()(uint32_t key) const { return (uint64_t)key; }
};

template <>
class KeyHash<int32_t> {
public:
    uint64_t operator()(int32_t key) const { return (uint64_t)key; }
};

template <>
class KeyHash<uint16_t> {
public:
    uint64_t operator()(uint16_t key) const { return (uint64_t)key; }
};

template <>
class KeyHash<int16_t> {
public:
    uint64_t operator()(int16_t key) const { return (uint64_t)key; }
};

template <>
class KeyHash<uint8_t> {
public:
    uint64_t operator()(uint8_t key) const { return (uint64_t)key; }
};

template <>
class KeyHash<int8_t> {
public:
    uint64_t operator()(int8_t key) const { return (uint64_t)key; }
};

template <>
class KeyHash<float> {
public:
    uint64_t operator()(float key) const {
        union {
            float k;
            uint64_t i;
        } a;

        a.k = a.i = 0;
        a.k = key;
        return a.i;
    }
};

template <>
class KeyHash<uint128_t> {
public:
    uint64_t operator()(uint128_t key) const { return key.value[0]; }
};

class HashUtil {
public:
    template <typename T>
    static size_t calculateHashValue(const T &data);

    template <class T>
    static size_t combineHash2(size_t seed, const T &v) {
        size_t seed2 = calculateHashValue(v);
        return Hash128to64(uint128(seed, seed2));
    }

    template <class T>
    static void combineHash(size_t &seed, const T &v) {
        seed = combineHash2(seed, v);
    }

    template <typename Arg, typename... Args>
    static size_t hashAll(const Arg &arg, const Args &...args) {
        return combineHash2(calculateHashValue(arg), hashAll(args...));
    }

    static size_t hashAll() { return 0; }
};

template <>
inline void HashUtil::combineHash(size_t &seed, const size_t &v) {
    seed = Hash128to64(uint128(seed, v));
}

template <typename T>
size_t HashUtil::calculateHashValue(const T &data) {
    std::hash<T> hasher;
    return hasher(data);
}

template <>
inline size_t HashUtil::calculateHashValue(const autil::StringView &data) {
    return CityHash64(data.data(), data.size());
}

template <>
inline size_t HashUtil::calculateHashValue(const autil::MultiChar &data) {
    return CityHash64(data.data(), data.size());
}

template <>
inline size_t HashUtil::calculateHashValue(const std::string &data) {
    return CityHash64(data.data(), data.size());
}

#define CALCULATE_HASH_VALUE(Type)                                                                                     \
    template <>                                                                                                        \
    inline size_t HashUtil::calculateHashValue<Type>(const Type &data) {                                               \
        return CityHash64((char *)&data, sizeof(Type));                                                                \
    }

#define CALCULATE_HASH_VALUE_SMALL_INT(Type)                                                                           \
    template <>                                                                                                        \
    inline size_t HashUtil::calculateHashValue<Type>(const Type &data) {                                               \
        int64_t tmp = data;                                                                                            \
        return CityHash64((char *)&tmp, sizeof(int64_t));                                                              \
    }

#define CALCULATE_HASH_VALUE_SMALL_UINT(Type)                                                                          \
    template <>                                                                                                        \
    inline size_t HashUtil::calculateHashValue<Type>(const Type &data) {                                               \
        uint64_t tmp = data;                                                                                           \
        return CityHash64((char *)&tmp, sizeof(uint64_t));                                                             \
    }

CALCULATE_HASH_VALUE_SMALL_INT(int8_t)
CALCULATE_HASH_VALUE_SMALL_INT(int16_t)
CALCULATE_HASH_VALUE_SMALL_INT(int32_t)
CALCULATE_HASH_VALUE_SMALL_UINT(uint8_t)
CALCULATE_HASH_VALUE_SMALL_UINT(uint16_t)
CALCULATE_HASH_VALUE_SMALL_UINT(uint32_t)
CALCULATE_HASH_VALUE(int64_t)
CALCULATE_HASH_VALUE(uint64_t)
CALCULATE_HASH_VALUE(float)
CALCULATE_HASH_VALUE(double)

#undef CALCULATE_HASH_VALUE
#undef CALCULATE_HASH_VALUE_SMALL_INT
#undef CALCULATE_HASH_VALUE_SMALL_UINT

#define CALCULATE_MULTI_VALUE_HASH_VALUE(Type)                                                                         \
    template <>                                                                                                        \
    inline size_t HashUtil::calculateHashValue<Type>(const Type &data) {                                               \
        if (data.size() == 0) {                                                                                        \
            return CityHash64(nullptr, 0);                                                                             \
        }                                                                                                              \
        return CityHash64(data.getBaseAddress(), data.length());                                                       \
    }
CALCULATE_MULTI_VALUE_HASH_VALUE(autil::MultiUInt8);
CALCULATE_MULTI_VALUE_HASH_VALUE(autil::MultiInt8);
CALCULATE_MULTI_VALUE_HASH_VALUE(autil::MultiUInt16);
CALCULATE_MULTI_VALUE_HASH_VALUE(autil::MultiInt16);
CALCULATE_MULTI_VALUE_HASH_VALUE(autil::MultiUInt32);
CALCULATE_MULTI_VALUE_HASH_VALUE(autil::MultiInt32);
CALCULATE_MULTI_VALUE_HASH_VALUE(autil::MultiUInt64);
CALCULATE_MULTI_VALUE_HASH_VALUE(autil::MultiInt64);
CALCULATE_MULTI_VALUE_HASH_VALUE(autil::MultiFloat);
CALCULATE_MULTI_VALUE_HASH_VALUE(autil::MultiDouble);
CALCULATE_MULTI_VALUE_HASH_VALUE(autil::MultiString);

#undef CALCULATE_MULTI_VALUE_HASH_VALUE

} // namespace autil
