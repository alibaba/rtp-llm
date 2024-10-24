#include <google/protobuf/rawstring.h>
#include <google/protobuf/rawstring_compatible_unittest.pb.h>
#include <google/protobuf/testing/googletest.h>
#include <gtest/gtest.h>
#include <string.h>
#include <vector>

namespace google {
namespace protobuf {

void CheckStringMessageOptional(StringMessage& message,
                                bool has_optional_string,
                                const std::string& optional_string,
                                bool has_optional_bytes,
                                const std::string& optional_bytes) {
    EXPECT_EQ(has_optional_string, message.has_optional_string_value());
    EXPECT_EQ(optional_string, message.optional_string_value());

    EXPECT_EQ(has_optional_bytes, message.has_optional_bytes_value());
    EXPECT_EQ(optional_bytes, message.optional_bytes_value());
}

void CheckStringMessageRequired(StringMessage& message,
                                const std::string& required_string,
                                const std::string& required_bytes) {
    EXPECT_EQ(required_string, message.required_string_value());
    EXPECT_EQ(required_bytes, message.required_bytes_value());
}

void CheckStringMessageRepeated(StringMessage& message,
                                const std::vector<std::string>& repeated_string,
                                const std::vector<std::string>& repeated_bytes) {
    EXPECT_EQ(repeated_string.size(), message.repeated_string_value_size());
    for (size_t i = 0; i < repeated_string.size(); i ++) {
        EXPECT_EQ(repeated_string[i], message.repeated_string_value(i));
    } 

    EXPECT_EQ(repeated_bytes.size(), message.repeated_bytes_value_size());
    for (size_t i = 0; i < repeated_bytes.size(); i ++) {
        EXPECT_EQ(repeated_bytes[i], message.repeated_bytes_value(i));
    }
}

void CheckStringMessageOneof(StringMessage& message,
                             bool has_oneof_string,
                             const std::string& oneof_string,
                             bool has_oneof_bytes,
                             const std::string& oneof_bytes) {
    EXPECT_EQ(has_oneof_string, message.has_oneof_string_value());
    EXPECT_EQ(oneof_string, message.oneof_string_value());

    EXPECT_EQ(has_oneof_bytes, message.has_oneof_bytes_value());
    EXPECT_EQ(oneof_bytes, message.oneof_bytes_value());
}

void CheckRawStringMessageOptional(RawStringMessage& message,
                                bool has_optional_string,
                                const std::string& optional_string,
                                bool has_optional_bytes,
                                const std::string& optional_bytes) {
    EXPECT_EQ(has_optional_string, message.has_optional_string_value());
    EXPECT_EQ(optional_string, message.optional_string_value().to_string());

    EXPECT_EQ(has_optional_bytes, message.has_optional_bytes_value());
    EXPECT_EQ(optional_bytes, message.optional_bytes_value().to_string());
}

void CheckRawStringMessageRequired(RawStringMessage& message,
                                const std::string& required_string,
                                const std::string& required_bytes) {
    EXPECT_EQ(required_string, message.required_string_value().to_string());
    EXPECT_EQ(required_bytes, message.required_bytes_value().to_string());
}

void CheckRawStringMessageRepeated(RawStringMessage& message,
                                const std::vector<std::string>& repeated_string,
                                const std::vector<std::string>& repeated_bytes) {
    EXPECT_EQ(repeated_string.size(), message.repeated_string_value_size());
    for (size_t i = 0; i < repeated_string.size(); i ++) {
        EXPECT_EQ(repeated_string[i], message.repeated_string_value(i).to_string());
    } 

    EXPECT_EQ(repeated_bytes.size(), message.repeated_bytes_value_size());
    for (size_t i = 0; i < repeated_bytes.size(); i ++) {
        EXPECT_EQ(repeated_bytes[i], message.repeated_bytes_value(i).to_string());
    }
}

void CheckRawStringMessageOneof(RawStringMessage& message,
                             bool has_oneof_string,
                             const std::string& oneof_string,
                             bool has_oneof_bytes,
                             const std::string& oneof_bytes) {
    EXPECT_EQ(has_oneof_string, message.has_oneof_string_value());
    EXPECT_EQ(oneof_string, message.oneof_string_value().to_string());

    EXPECT_EQ(has_oneof_bytes, message.has_oneof_bytes_value());
    EXPECT_EQ(oneof_bytes, message.oneof_bytes_value().to_string());
}

TEST(RawStringCompatibleTest, TestFromRawString_SerializeString_Empty) {
    RawStringMessage rawstring_message;
    CheckRawStringMessageOptional(rawstring_message, false, "", false, "");
    CheckRawStringMessageRequired(rawstring_message, "", "");
    CheckRawStringMessageRepeated(rawstring_message, {}, {});
    CheckRawStringMessageOneof(rawstring_message, false, "", false, "");

    rawstring_message.set_required_string_value("");
    rawstring_message.set_required_bytes_value("");
    CheckRawStringMessageRequired(rawstring_message, "", "");

    std::string value = rawstring_message.SerializeAsString();

    StringMessage string_message;
    EXPECT_TRUE(string_message.ParseFromString(value));

    CheckStringMessageOptional(string_message, false, "", false, "");
    CheckStringMessageRequired(string_message, "", "");
    CheckStringMessageRepeated(string_message, {}, {});
    CheckStringMessageOneof(string_message, false, "", false, "");
}

TEST(RawStringCompatibleTest, TestFromRawString_SerializeString_Value) {
    RawStringMessage rawstring_message;
    CheckRawStringMessageOptional(rawstring_message, false, "", false, "");
    CheckRawStringMessageRequired(rawstring_message, "", "");
    CheckRawStringMessageRepeated(rawstring_message, {}, {});
    CheckRawStringMessageOneof(rawstring_message, false, "", false, "");

    rawstring_message.set_optional_string_value("optional string");
    rawstring_message.set_optional_bytes_value("optional bytes");
    rawstring_message.set_required_string_value("required string");
    rawstring_message.set_required_bytes_value("required bytes");
    rawstring_message.add_repeated_string_value("repeated string 1");
    rawstring_message.add_repeated_string_value("repeated string 2");
    rawstring_message.add_repeated_bytes_value("repeated bytes 1");
    rawstring_message.add_repeated_bytes_value("repeated bytes 2");
    rawstring_message.set_oneof_string_value("oneof string");
    rawstring_message.set_oneof_bytes_value("oneof bytes");

    CheckRawStringMessageOptional(rawstring_message, true, "optional string", true, "optional bytes");
    CheckRawStringMessageRequired(rawstring_message, "required string", "required bytes");
    CheckRawStringMessageRepeated(rawstring_message, {"repeated string 1", "repeated string 2"}, {"repeated bytes 1", "repeated bytes 2"});
    CheckRawStringMessageOneof(rawstring_message, true, "oneof string", true, "oneof bytes");

    std::string value = rawstring_message.SerializeAsString();

    StringMessage string_message;
    EXPECT_TRUE(string_message.ParseFromString(value));

    CheckStringMessageOptional(string_message, true, "optional string", true, "optional bytes");
    CheckStringMessageRequired(string_message, "required string", "required bytes");
    CheckStringMessageRepeated(string_message, {"repeated string 1", "repeated string 2"}, {"repeated bytes 1", "repeated bytes 2"});
    CheckStringMessageOneof(string_message, true, "oneof string", true, "oneof bytes");
}

TEST(RawStringCompatibleTest, TestToRawString_SerializeString_Empty) {
    StringMessage string_message;
    CheckStringMessageOptional(string_message, false, "", false, "");
    CheckStringMessageRequired(string_message, "", "");
    CheckStringMessageRepeated(string_message, {}, {});
    CheckStringMessageOneof(string_message, false, "", false, "");

    string_message.set_required_string_value("");
    string_message.set_required_bytes_value("");
    CheckStringMessageRequired(string_message, "", "");

    std::string value = string_message.SerializeAsString();

    RawStringMessage rawstring_message;
    EXPECT_TRUE(rawstring_message.ParseFromString(value));

    CheckRawStringMessageOptional(rawstring_message, false, "", false, "");
    CheckRawStringMessageRequired(rawstring_message, "", "");
    CheckRawStringMessageRepeated(rawstring_message, {}, {});
    CheckRawStringMessageOneof(rawstring_message, false, "", false, "");
}

TEST(RawStringCompatibleTest, TestToRawString_SerializeString_Value) {
    StringMessage string_message;
    CheckStringMessageOptional(string_message, false, "", false, "");
    CheckStringMessageRequired(string_message, "", "");
    CheckStringMessageRepeated(string_message, {}, {});
    CheckStringMessageOneof(string_message, false, "", false, "");

    string_message.set_optional_string_value("optional string");
    string_message.set_optional_bytes_value("optional bytes");
    string_message.set_required_string_value("required string");
    string_message.set_required_bytes_value("required bytes");
    string_message.add_repeated_string_value("repeated string 1");
    string_message.add_repeated_string_value("repeated string 2");
    string_message.add_repeated_bytes_value("repeated bytes 1");
    string_message.add_repeated_bytes_value("repeated bytes 2");
    string_message.set_oneof_string_value("oneof string");
    string_message.set_oneof_bytes_value("oneof bytes");

    CheckStringMessageOptional(string_message, true, "optional string", true, "optional bytes");
    CheckStringMessageRequired(string_message, "required string", "required bytes");
    CheckStringMessageRepeated(string_message, {"repeated string 1", "repeated string 2"}, {"repeated bytes 1", "repeated bytes 2"});
    CheckStringMessageOneof(string_message, true, "oneof string", true, "oneof bytes");

    std::string value = string_message.SerializeAsString();
    
    RawStringMessage rawstring_message;
    EXPECT_TRUE(rawstring_message.ParseFromString(value));

    CheckRawStringMessageOptional(rawstring_message, true, "optional string", true, "optional bytes");
    CheckRawStringMessageRequired(rawstring_message, "required string", "required bytes");
    CheckRawStringMessageRepeated(rawstring_message, {"repeated string 1", "repeated string 2"}, {"repeated bytes 1", "repeated bytes 2"});
    CheckRawStringMessageOneof(rawstring_message, true, "oneof string", true, "oneof bytes");
}

}
} 