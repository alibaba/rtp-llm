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

#include <google/protobuf/test_util_rawstring_lite.h>
#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <gtest/gtest.h>

namespace google {
namespace protobuf {

const std::string kOptionalRawStringSetValue = "128";
const std::string kRepeatedRawStringSetValue0 = "228";
const std::string kRepeatedRawStringSetValue1 = "328";
const std::string kDefaultRawStringSetValue = "428";
const std::string kRepeatedRawStringModifiedValue1 = "528";
const std::string kDefaultRawStringOriginValue = "qwe";
const std::string kOneof1RawStringSetValue = "104";
const std::string kOneof2RawStringSetValue = "204";

void TestUtilLiteRawString::SetAllFields(::protobuf_unittest::TestAllTypesLite* message) {
  message->set_optional_rawstring(kOptionalRawStringSetValue);
  message->add_repeated_rawstring(kRepeatedRawStringSetValue0);
  message->add_repeated_rawstring(kRepeatedRawStringSetValue1);
  message->set_default_rawstring(kDefaultRawStringSetValue);
}

void TestUtilLiteRawString::ExpectAllFieldsSet(
    const ::protobuf_unittest::TestAllTypesLite& message) {
  EXPECT_TRUE(message.has_optional_rawstring());
  EXPECT_EQ(kOptionalRawStringSetValue, message.optional_rawstring().to_string());

  ASSERT_EQ(2, message.repeated_rawstring_size());
  EXPECT_EQ(kRepeatedRawStringSetValue0, message.repeated_rawstring(0).to_string());
  EXPECT_EQ(kRepeatedRawStringSetValue1, message.repeated_rawstring(1).to_string());

  EXPECT_TRUE(message.has_default_rawstring());
  EXPECT_EQ(kDefaultRawStringSetValue, message.default_rawstring().to_string());
}

void TestUtilLiteRawString::ModifyRepeatedFields(
    ::protobuf_unittest::TestAllTypesLite* message) {
  message->set_repeated_rawstring(1, kRepeatedRawStringModifiedValue1);
}

void TestUtilLiteRawString::ExpectRepeatedFieldsModified(
    const ::protobuf_unittest::TestAllTypesLite& message) {
  ASSERT_EQ(2, message.repeated_rawstring_size());
  EXPECT_EQ(kRepeatedRawStringSetValue0, message.repeated_rawstring(0).to_string());
  EXPECT_EQ(kRepeatedRawStringModifiedValue1, message.repeated_rawstring(1).to_string());
}

void TestUtilLiteRawString::ExpectClear(
    const ::protobuf_unittest::TestAllTypesLite& message) {
  EXPECT_FALSE(message.has_optional_rawstring());
  EXPECT_EQ(0, message.optional_rawstring().size());

  EXPECT_EQ(0, message.repeated_rawstring_size());

  EXPECT_FALSE(message.has_default_rawstring());
  EXPECT_EQ(kDefaultRawStringOriginValue, message.default_rawstring().to_string());
}

void TestUtilLiteRawString::SetAllExtensions(
    ::protobuf_unittest::TestAllExtensionsLite* message) {
  message->SetExtension(::protobuf_unittest::optional_rawstring_extension_lite, kOptionalRawStringSetValue);
  message->AddExtension(::protobuf_unittest::repeated_rawstring_extension_lite, kRepeatedRawStringSetValue0);
  message->AddExtension(::protobuf_unittest::repeated_rawstring_extension_lite, kRepeatedRawStringSetValue1);
  message->SetExtension(::protobuf_unittest::default_rawstring_extension_lite, kDefaultRawStringSetValue);
}

void TestUtilLiteRawString::ExpectAllExtensionsSet(
    const ::protobuf_unittest::TestAllExtensionsLite& message) {
  EXPECT_TRUE(message.HasExtension(::protobuf_unittest::optional_rawstring_extension_lite));
  EXPECT_EQ(kOptionalRawStringSetValue, message.GetExtension(::protobuf_unittest::optional_rawstring_extension_lite));
  
  ASSERT_EQ(2, message.ExtensionSize(::protobuf_unittest::repeated_rawstring_extension_lite));
  EXPECT_EQ(kRepeatedRawStringSetValue0,
            message.GetExtension(::protobuf_unittest::repeated_rawstring_extension_lite, 0));
  EXPECT_EQ(kRepeatedRawStringSetValue1,
            message.GetExtension(::protobuf_unittest::repeated_rawstring_extension_lite, 1));

  EXPECT_TRUE(message.HasExtension(::protobuf_unittest::default_rawstring_extension_lite));
  EXPECT_EQ(kDefaultRawStringSetValue,
            message.GetExtension(::protobuf_unittest::default_rawstring_extension_lite));
}

void TestUtilLiteRawString::ModifyRepeatedExtensions(
    ::protobuf_unittest::TestAllExtensionsLite* message) {
  message->SetExtension(::protobuf_unittest::repeated_rawstring_extension_lite, 1, kRepeatedRawStringModifiedValue1);
}

void TestUtilLiteRawString::ExpectRepeatedExtensionsModified(
    const ::protobuf_unittest::TestAllExtensionsLite& message) {
  ASSERT_EQ(2, message.ExtensionSize(::protobuf_unittest::repeated_rawstring_extension_lite));
  EXPECT_EQ(kRepeatedRawStringSetValue0,
            message.GetExtension(::protobuf_unittest::repeated_rawstring_extension_lite, 0));
  EXPECT_EQ(kRepeatedRawStringModifiedValue1,
            message.GetExtension(::protobuf_unittest::repeated_rawstring_extension_lite, 1));
}

void TestUtilLiteRawString::ExpectExtensionsClear(
    const ::protobuf_unittest::TestAllExtensionsLite& message) {
  EXPECT_FALSE(message.HasExtension(::protobuf_unittest::optional_rawstring_extension_lite));
  EXPECT_EQ(0, message.GetExtension(::protobuf_unittest::optional_rawstring_extension_lite).size());

  EXPECT_EQ(0, message.ExtensionSize(::protobuf_unittest::repeated_rawstring_extension_lite));

  EXPECT_FALSE(message.HasExtension(::protobuf_unittest::default_rawstring_extension_lite));  
  EXPECT_EQ(kDefaultRawStringOriginValue,
            message.GetExtension(::protobuf_unittest::default_rawstring_extension_lite));
}

}
}