#pragma once

#include "navi/common.h"
#include "autil/legacy/jsonizable.h"

namespace navi {

class SummaryTableColumn : public autil::legacy::Jsonizable {
public:
    SummaryTableColumn() {
    }
    SummaryTableColumn(const std::string &name)
        : _name(name)
    {
    }
    ~SummaryTableColumn() {
    }
public:
    template <typename T>
    void setData(size_t index, const T &t) {
        if (_datas.size() <= index) {
            _datas.resize(index + 1);
        }
        _datas[index] = t;
    }
    template <typename T>
    void addData(const T &t) {
        _datas.push_back(t);
    }
public:
    void setDesc(const std::string &desc) {
        _desc = desc;
    }
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override {
        json.Jsonize("name", _name, _name);
        json.Jsonize("desc", _desc, _desc);
        json.Jsonize("data", _datas, _datas);
    }
private:
    std::string _name;
    std::string _desc;
    std::vector<autil::legacy::Any> _datas;
};

NAVI_TYPEDEF_PTR(SummaryTableColumn);

#define TABLE_ADD_COLUMN_VALUE(TABLE, COLUMN, VALUE) (TABLE)->addColumn(COLUMN)->addData(VALUE);
#define TABLE_ADD_VALUE(COLUMN, VALUE) (COLUMN)->addData(VALUE);

class SummaryTableBuilder : public autil::legacy::Jsonizable {
public:
    SummaryTableBuilder() {
    }
    SummaryTableBuilder(const std::string &name)
        : _name(name)
    {
    }
    ~SummaryTableBuilder() {
    }
public:
    void setDesc(const std::string &desc) {
        _desc = desc;
    }
    SummaryTableColumnPtr addColumn(const std::string &name);
    SummaryTableColumnPtr getColumn(size_t index) const {
        if (index < _columnVec.size()) {
            return _columnVec[index];
        } else {
            return nullptr;
        }
    }
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override {
        json.Jsonize("name", _name, _name);
        json.Jsonize("desc", _desc, _desc);
        json.Jsonize("columns", _columnVec, _columnVec);
    }
private:
    std::string _name;
    std::string _desc;
    std::vector<SummaryTableColumnPtr> _columnVec;
    std::map<std::string, SummaryTableColumnPtr> _columnMap;
};

NAVI_TYPEDEF_PTR(SummaryTableBuilder);

class SummaryBuilder : public autil::legacy::Jsonizable {
public:
    SummaryBuilder() {
    }
    ~SummaryBuilder() {
    }
public:
    void setDesc(const std::string &desc) {
        _desc = desc;
    }
    SummaryTableBuilderPtr newTable(const std::string &name);
    std::string toString() const;
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override {
        json.Jsonize("version", _version, _version);
        json.Jsonize("desc", _desc, _desc);
        json.Jsonize("tables", _tableVec, _tableVec);
    }
private:
    std::string _version = "1.0.0";
    std::string _desc;
    std::vector<SummaryTableBuilderPtr> _tableVec;
    std::map<std::string, SummaryTableBuilderPtr> _tableMap;
};

}

