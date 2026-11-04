#include "navi/engine/StringHashTable.h"

namespace navi {

StringHashTable::StringHashTable() {
}

StringHashTable::~StringHashTable() {
}

size_t StringHashTable::getHash(const std::string &str) {
    auto value = _hashFunc(str);
    auto tableMapDef = _fullTable.mutable_table();
    auto it = tableMapDef->find(value);
    if (tableMapDef->end() == it) {
        auto incTableMapDef = _incTable.mutable_table();
        if (incTableMapDef->end() == incTableMapDef->find(value)) {
            (*incTableMapDef)[value] = str;
        }
    }
    return value;
}

void StringHashTable::toProto(SymbolTableDef *symbolTableDef) {
    symbolTableDef->MergeFrom(_incTable);
    _fullTable.MergeFrom(_incTable);
    _incTable.Clear();
}

void StringHashTable::fromStr(const std::string &str) {
    _incTable.ParseFromString(str);
}

}

