#pragma once

#include "navi/common.h"
#include "navi/proto/GraphVis.pb.h"

namespace navi {

class StringHashTable
{
public:
    StringHashTable();
    ~StringHashTable();
    StringHashTable(const StringHashTable &) = delete;
    StringHashTable &operator=(const StringHashTable &) = delete;
public:
    size_t getHash(const std::string &str);
    bool needToProto() const {
        return (_incTable.table_size() > 0);
    }
    void toProto(SymbolTableDef *symbolTableDef);
    void fromStr(const std::string &str);
public:
    std::hash<std::string> _hashFunc;
    SymbolTableDef _fullTable;
    SymbolTableDef _incTable;
};

NAVI_TYPEDEF_PTR(StringHashTable);

}

