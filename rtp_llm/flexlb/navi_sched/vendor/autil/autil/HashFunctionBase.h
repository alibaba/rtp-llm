#pragma once

#include <assert.h>
#include <memory>
#include <set>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace autil {

class HashFunctionBase {
public:
    HashFunctionBase(const std::string &hashFunction, uint32_t hashSize = 65536)
        : _hashFunction(hashFunction), _hashSize(hashSize) {}
    virtual ~HashFunctionBase() {}

public:
    virtual bool init() { return true; }
    virtual uint32_t getHashId(const std::string &str) const = 0;
    virtual uint32_t getHashId(const char *buf, size_t len) const {
        assert(buf != NULL);
        return getHashId(std::string(buf, len));
    }

public:
    virtual uint32_t getHashId(const std::vector<std::string> &strVec) const;
    virtual std::vector<std::pair<uint32_t, uint32_t>> getHashRange(const std::vector<std::string> &strVec) const;
    virtual std::vector<std::pair<uint32_t, uint32_t>> getHashRange(uint32_t partId) const;
    virtual std::vector<std::pair<uint32_t, uint32_t>> getHashRange(uint32_t partId, float ratio) const;

    std::string getFunctionName() const;
    uint32_t getHashSize() const;
    const std::set<std::string> &getCanEmptyFields() const { return _canEmptyFields; }

protected:
    std::string _hashFunction;
    uint32_t _hashSize;
    std::set<std::string> _canEmptyFields;
};
typedef std::shared_ptr<HashFunctionBase> HashFunctionBasePtr;

constexpr char HASH_FUNC_DOCID[] = "DOCID";
constexpr char HASH_FUNC_HASH[] = "HASH";

} // namespace autil
