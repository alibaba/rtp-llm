#pragma once

#include <memory>

#include "autil/legacy/jsonizable.h"
#include "google/protobuf/message.h"
namespace http_arpc {

class ProtoJsonizer {
public:
    static const std::string ORIGINAL_STRING;
    static const std::string HTTP_STATUS_CODE;

public:
    ProtoJsonizer();
    virtual ~ProtoJsonizer();

private:
    ProtoJsonizer(const ProtoJsonizer &);
    ProtoJsonizer &operator=(const ProtoJsonizer &);

public:
    static bool fromJsonString(const std::string &jsonStr, ::google::protobuf::Message *message);
    static std::string toJsonString(const ::google::protobuf::Message &message, bool outputDefault = false);
    static void fromJsonMap(const autil::legacy::json::JsonMap &jsonMap, ::google::protobuf::Message *message);
    static void toJsonMap(const ::google::protobuf::Message &message,
                          autil::legacy::json::JsonMap &jsonMap,
                          bool outputDefault = false);

    static std::string toJsonString(const ::google::protobuf::Message &message,
                                    const std::string &responseField,
                                    bool outputDefault = false);

public:
    virtual bool fromJson(const std::string &jsonStr,
                          int32_t httpMethod,
                          const std::string &uri, //only used when post
                          const std::map<std::string, std::string> &headers,
                          ::google::protobuf::Message *message);
    virtual bool fromJson(const std::string &jsonStr, ::google::protobuf::Message *message);
    virtual std::string toJson(const ::google::protobuf::Message &message);
    virtual int getStatusCode(const google::protobuf::Message &message);

public:
    void setResponseField(const std::string &responseField);

protected:
    static void setOneValue(const ::google::protobuf::FieldDescriptor *fieldDesc,
                            const autil::legacy::Any &any,
                            ::google::protobuf::Message *message);
    template <typename T>
    static T parseType(const autil::legacy::Any &any);

private:
    std::string _responseField;
};

typedef std::shared_ptr<ProtoJsonizer> ProtoJsonizerPtr;
} // namespace http_arpc
