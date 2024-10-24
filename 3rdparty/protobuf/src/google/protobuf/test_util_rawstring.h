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

#ifndef GOOGLE_PROTOBUF_TEST_UTIL_RAW_STRING_H__
#define GOOGLE_PROTOBUF_TEST_UTIL_RAW_STRING_H__

#include <google/protobuf/unittest.pb.h>
#include <google/protobuf/unittest_lite.pb.h>

namespace google {
namespace protobuf {

class TestUtilRawString {
public:
    static void SetAllFields(::protobuf_unittest::TestAllTypes* message);
    static void SetAllExtensions(::protobuf_unittest::TestAllExtensions* message);
    static void SetOneof1(::protobuf_unittest::TestOneof2* message);
    static void SetOneof2(::protobuf_unittest::TestOneof2* message);

    static void ModifyRepeatedFields(::protobuf_unittest::TestAllTypes* message);
    static void ModifyRepeatedExtensions(::protobuf_unittest::TestAllExtensions* message);

    static void ExpectAllFieldsSet(const ::protobuf_unittest::TestAllTypes& message);
    static void ExpectAllExtensionsSet(const ::protobuf_unittest::TestAllExtensions& message);
    static void ExpectOneofSet1(const ::protobuf_unittest::TestOneof2& message);
    static void ExpectOneofSet2(const ::protobuf_unittest::TestOneof2& message);

    static void ExpectRepeatedFieldsModified(const ::protobuf_unittest::TestAllTypes& message);
    static void ExpectRepeatedExtensionsModified(const ::protobuf_unittest::TestAllExtensions& message);

    static void ExpectClear(const ::protobuf_unittest::TestAllTypes& message);
    static void ExpectExtensionsClear(const ::protobuf_unittest::TestAllExtensions& message);
    static void ExpectOneofClear(const ::protobuf_unittest::TestOneof2& message);

    static void ExpectLastRepeatedsRemoved(const ::protobuf_unittest::TestAllTypes& message);
    static void ExpectLastRepeatedExtensionsRemoved(const ::protobuf_unittest::TestAllExtensions& message);

    static void ExpectRepeatedsSwapped(const ::protobuf_unittest::TestAllTypes& message);
    static void ExpectRepeatedExtensionsSwapped(const ::protobuf_unittest::TestAllExtensions& message);
};
}
}


#endif // GOOGLE_PROTOBUF_TEST_UTIL_RAW_STRING_H__