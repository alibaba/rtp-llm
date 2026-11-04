#pragma once

#include "autil/NoCopyable.h"
#include "navi/common.h"
#include "navi/engine/TypeInfo.h"
#include "navi/engine/SummaryBuilder.h"
#include "autil/legacy/jsonizable.h"

namespace navi {

class Type;
class Data;
NAVI_TYPEDEF_PTR(Data);

class Data : public autil::NoCopyable
{
public:
    Data(const std::string &typeId);
    virtual ~Data();
public:
    // do not use this invalid ctor
    Data(const char *) = delete;
private:
    Data(const Data &);
    Data& operator=(const Data &);
public:
    const std::string &getTypeId() const {
        return _typeId;
    }
    virtual SummaryErrorCode getSummary(const std::string &query, SummaryBuilder &builder) const {
        return SEC_NOT_SUPPORT;
    }
    virtual void show() const {
    }
    // return nullptr if no need clone
    virtual DataPtr clone() const {
        return DataPtr();
    }
    virtual Data *pointerClone() const {
        return nullptr;
    }
private:
    const std::string &_typeId;
};

struct StreamData {
public:
    StreamData() {
        reset();
    }
    void reset(bool clearEof = true) {
        hasValue = false;
        data.reset();
        if (dataPointer) {
            delete dataPointer;
        }
        dataPointer = nullptr;
        if (clearEof) {
            eof = false;
        }
    }
    bool isDataPointer() const {
        return dataPointer || !data;
    }
    Data *stealDataPointer() const {
        if (dataPointer) {
            auto tmp = dataPointer;
            dataPointer = nullptr;
            return tmp;
        } else {
            return nullptr;
        }
    }
public:
    bool hasValue;
    bool eof;
    DataPtr data;
private:
    mutable Data *dataPointer = nullptr;
private:
    friend class Node;
};

#define BIND_DATA_TO(member)                                                   \
    [](void *obj,                                                              \
       void *castObj,                                                          \
       navi::Data *namedData,                                                  \
       int64_t *objOffset) -> bool {                                           \
        if (unlikely(nullptr != objOffset)) {                                  \
            using T =                                                          \
                std::remove_const_t<std::remove_pointer_t<decltype(this)>>;    \
            CAST_OFFSET_OBJ();                                                 \
            return false;                                                      \
        }                                                                      \
        using T = std::remove_const_t<std::remove_pointer_t<decltype(this)>>;  \
        T *typedObj = navi_bind_cast<T>(obj);                                  \
        typedObj->member = nullptr;                                            \
        if (!namedData) {                                                      \
            return true;                                                       \
        }                                                                      \
        auto *typedData = dynamic_cast<decltype(typedObj->member)>(namedData); \
        typedObj->member = typedData;                                          \
        auto ret = (typedData != nullptr);                                     \
        if (!ret) {                                                            \
            NAVI_KERNEL_LOG(                                                   \
                ERROR,                                                         \
                "bind named data failed, can't cast type [%s] to type [%s], "  \
                "class: [%s], member: [%s], file: [%s:%d]",                    \
                typeid(*namedData).name(),                                     \
                typeid(decltype(typedObj->member)).name(),                     \
                typeid(T).name(),                                              \
                #member,                                                       \
                __FILE__,                                                      \
                __LINE__);                                                     \
        }                                                                      \
        return ret;                                                            \
    }
}
