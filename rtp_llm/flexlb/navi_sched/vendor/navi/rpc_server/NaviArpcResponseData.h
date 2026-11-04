#pragma once

#include "navi/engine/Data.h"
#include "navi/engine/Type.h"

namespace navi {

class NaviArpcResponseData : public Data
{
public:
    NaviArpcResponseData(google::protobuf::Message *response);
    ~NaviArpcResponseData();
    NaviArpcResponseData(const NaviArpcResponseData &) = delete;
    NaviArpcResponseData &operator=(const NaviArpcResponseData &) = delete;
public:
    static const std::string TYPE_ID;
public:
    const google::protobuf::Message *getResponse() const {
        return _response;
    }
    google::protobuf::Message *stealResponse();
    virtual SummaryErrorCode getSummary(const std::string &query, SummaryBuilder &builder) const override;
    void setQueryKey(const std::string &key) {
        _queryKey = key;
    }
    const std::string &getQueryKey() const {
        return _queryKey;
    }
private:
    google::protobuf::Message *_response;
    std::string _queryKey;
};

class NaviArpcResponseType : public Type {
public:
    NaviArpcResponseType();
    ~NaviArpcResponseType();
public:
    navi::TypeErrorCode serialize(navi::TypeContext &ctx, const navi::DataPtr &data) const override;
    navi::TypeErrorCode deserialize(navi::TypeContext &ctx, navi::DataPtr &data) const override;
};

}

