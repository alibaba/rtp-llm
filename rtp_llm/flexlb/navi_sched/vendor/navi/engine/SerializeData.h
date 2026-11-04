#pragma once

#include "navi/common.h"
#include "navi/engine/Data.h"
#include "navi/log/NaviLogger.h"
#include "navi/proto/NaviStream.pb.h"
#include "autil/Lock.h"

namespace navi {

struct SerializeData {
public:
    SerializeData();
    ~SerializeData();
public:
    static bool serialize(const NaviObjectLogger &_logger,
                          NaviPartId toPartId,
                          NaviPartId fromPartId,
                          const DataPtr &data,
                          bool eof,
                          const Type *dataType,
                          NaviPortData &serializeData);
    static bool serializeToStr(const NaviObjectLogger *_logger,
                               const Type *dataType,
                               const DataPtr &data,
                               std::string &result);
    static bool deserialize(const NaviObjectLogger &_logger,
                            const NaviPortData &serializeData,
                            const Type *dataType,
                            DataPtr &dataPtr);
    static bool deserializeFromStr(const NaviObjectLogger *_logger,
                                   const std::string &dataStr,
                                   const Type *dataType,
                                   DataPtr &dataPtr);
private:
    static bool fillData(const NaviObjectLogger &_logger,
                         const DataPtr &data, const Type *dataType,
                         NaviPortData &serializeData);
};

}
