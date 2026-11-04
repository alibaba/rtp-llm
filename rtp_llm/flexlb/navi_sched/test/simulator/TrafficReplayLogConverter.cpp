#include "autil/StringUtil.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <cmath>
#include <fstream>
#include <string>

namespace {

const rapidjson::Value *findValue(const rapidjson::Value &object, const char *name)
{
    auto it = object.FindMember(name);
    return object.MemberEnd() == it ? nullptr : &it->value;
}

bool getInt64(const rapidjson::Value &object, const char *name, int64_t &value)
{
    auto field = findValue(object, name);
    if (!field) return false;
    if (field->IsInt64()) {
        value = field->GetInt64();
        return true;
    }
    return field->IsUint64() && field->GetUint64() <= INT64_MAX &&
           (value = static_cast<int64_t>(field->GetUint64()), true);
}

bool getTimestampMs(const rapidjson::Value &object, int64_t &value)
{
    auto field = findValue(object, "timestamp");
    if (!field) return false;
    if (field->IsDouble()) {
        value = static_cast<int64_t>(std::llround(field->GetDouble() * 1000.0));
        return value >= 0;
    }
    return getInt64(object, "timestamp", value);
}

bool writeRecord(const rapidjson::Value &source, const std::string &serviceName,
                 std::ofstream &output)
{
    int64_t timestampMs = 0;
    int64_t inputLength = 0;
    const auto blockHashes = findValue(source, "input_block_hash_ids");
    if (!getTimestampMs(source, timestampMs) ||
        !getInt64(source, "input_length", inputLength) || inputLength <= 0 ||
        !blockHashes || !blockHashes->IsArray()) {
        return false;
    }
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.Key("timestamp_ms");
    writer.Int64(timestampMs);
    writer.Key("service_name");
    writer.String(serviceName.c_str(), serviceName.size());
    writer.Key("input_length");
    writer.Int64(inputLength);
    writer.Key("input_block_hash_ids");
    writer.StartArray();
    for (const auto &value : blockHashes->GetArray()) {
        if (!value.IsString()) return false;
        uint64_t hash = 0;
        std::string hashString(value.GetString(), value.GetStringLength());
        if (!autil::StringUtil::hexStrToUint64(hashString.c_str(), hash)) {
            return false;
        }
        writer.String(hashString.c_str(), hashString.size());
    }
    writer.EndArray();
    writer.EndObject();
    output << buffer.GetString() << '\n';
    return output.good();
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 4) return 1;
    std::ifstream input(argv[1]);
    std::ofstream output(argv[2]);
    const std::string serviceName(argv[3]);
    if (!input.good() || !output.good() || serviceName.empty()) return 1;
    std::string line;
    while (std::getline(input, line)) {
        rapidjson::Document document;
        document.Parse(line.c_str(), line.size());
        auto services = document.HasParseError() || !document.IsObject()
                            ? nullptr
                            : findValue(document, "service_names");
        if (!services || !services->IsArray()) continue;
        for (const auto &value : services->GetArray()) {
            if (!value.IsString()) continue;
            if (serviceName == std::string(value.GetString(), value.GetStringLength()) &&
                !writeRecord(document, serviceName, output))
                return 1;
        }
    }
    return output.good() ? 0 : 1;
}
