// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <google/protobuf/test_util_rawstring_reflection.h>
#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <gtest/gtest.h>

namespace google {
namespace protobuf {

const std::string kOptiaonRawStringFieldName = "optional_rawstring";
const std::string kRepeatedRawStringFieldName = "repeated_rawstring";
const std::string kDefaultRawStringFieldName = "default_rawstring";
const std::string kOptionalRawStringSetValue = "128";
const std::string kRepeatedRawStringSetValue0 = "228";
const std::string kRepeatedRawStringSetValue1 = "328";
const std::string kDefaultRawStringSetValue = "428";
const std::string kRepeatedRawStringModifiedValue1 = "528";
const std::string kDefaultRawStringOriginValue = "qwe";
const std::string kOneof1RawStringSetValue = "104";
const std::string kOneof2RawStringSetValue = "204";

// Shorthand to get a FieldDescriptor for a field of TestAllTypes.
const FieldDescriptor* TestUtilReflectionRawString::F(
    const std::string& name) const {
  const FieldDescriptor* result = nullptr;
  if (base_descriptor_->name() == "TestAllExtensions" ||
      base_descriptor_->name() == "TestPackedExtensions") {
    result = base_descriptor_->file()->FindExtensionByName(name + "_extension");
  } else {
    result = base_descriptor_->FindFieldByName(name);
  }
  GOOGLE_CHECK(result != nullptr);
  return result;
}

void TestUtilReflectionRawString::SetAllFieldsViaReflection(
    Message *message) const {
  const Reflection* reflection = message->GetReflection();
  reflection->SetString(message, F(kOptiaonRawStringFieldName), kOptionalRawStringSetValue);
  reflection->AddString(message, F(kRepeatedRawStringFieldName), kRepeatedRawStringSetValue0);
  reflection->AddString(message, F(kRepeatedRawStringFieldName), kRepeatedRawStringSetValue1);
  reflection->SetString(message, F(kDefaultRawStringFieldName), kDefaultRawStringSetValue);
}

void TestUtilReflectionRawString::ModifyRepeatedFieldsViaReflection(
    Message* message) const {
  const Reflection* reflection = message->GetReflection();
  reflection->SetRepeatedString(message, F(kRepeatedRawStringFieldName), 1, kRepeatedRawStringModifiedValue1);
}

void TestUtilReflectionRawString::ExpectAllFieldsSetViaReflection(
    const Message& message) const {
  const Reflection* reflection = message.GetReflection();
  std::string raw_scratch;

  EXPECT_TRUE(reflection->HasField(message, F(kOptiaonRawStringFieldName)));
  EXPECT_EQ(kOptionalRawStringSetValue,
            reflection->GetString(
                message, F(kOptiaonRawStringFieldName)));
  EXPECT_EQ(kOptionalRawStringSetValue,
            reflection->GetStringReference(
                message, F(kOptiaonRawStringFieldName), &raw_scratch));

  ASSERT_EQ(2, reflection->FieldSize(message, F(kRepeatedRawStringFieldName)));
  EXPECT_EQ(kRepeatedRawStringSetValue0,
            reflection->GetRepeatedString(
                message, F(kRepeatedRawStringFieldName), 0));
  EXPECT_EQ(kRepeatedRawStringSetValue0,
            reflection->GetRepeatedStringReference(
                message, F(kRepeatedRawStringFieldName), 0, &raw_scratch));
  EXPECT_EQ(kRepeatedRawStringSetValue1,
            reflection->GetRepeatedString(
                message, F(kRepeatedRawStringFieldName), 1));
  EXPECT_EQ(kRepeatedRawStringSetValue1,
            reflection->GetRepeatedStringReference(
                message, F(kRepeatedRawStringFieldName), 1, &raw_scratch));
  
  EXPECT_TRUE(reflection->HasField(message, F(kDefaultRawStringFieldName)));
  EXPECT_EQ(kDefaultRawStringSetValue,
            reflection->GetString(message, F(kDefaultRawStringFieldName)));
  EXPECT_EQ(kDefaultRawStringSetValue,
            reflection->GetStringReference(
                message, F(kDefaultRawStringFieldName), &raw_scratch));
}

void TestUtilReflectionRawString::ExpectClearViaReflection(const Message& message) const {
  const Reflection* reflection = message.GetReflection();
  std::string raw_scratch;
  EXPECT_FALSE(reflection->HasField(message, F(kOptiaonRawStringFieldName)));
  EXPECT_EQ(0, reflection->GetString(message, F(kOptiaonRawStringFieldName)).size());
  EXPECT_EQ(0, reflection->GetStringReference(message, F(kOptiaonRawStringFieldName), &raw_scratch).size());

  EXPECT_EQ(0, reflection->FieldSize(message, F(kRepeatedRawStringFieldName)));

  EXPECT_FALSE(reflection->HasField(message, F(kDefaultRawStringFieldName)));
  EXPECT_EQ(kDefaultRawStringOriginValue,
            reflection->GetString(message, F(kDefaultRawStringFieldName)));
  EXPECT_EQ(kDefaultRawStringOriginValue,
            reflection->GetStringReference(
                message, F(kDefaultRawStringFieldName), &raw_scratch));
}

}
}