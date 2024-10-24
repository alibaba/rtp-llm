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

#include <google/protobuf/arenarawstring.h>

#include <gtest/gtest.h>

namespace google {
namespace protobuf {

using internal::ArenaRawStringPtr;

static std::RawString WrapRawString(const char *value) { return value; }

// Test ArenaRawStringPtr with arena == NULL.
TEST(ArenaRawStringPtrTest, ArenaRawStringPtrOnHeap) {
  ArenaRawStringPtr field;
  std::RawString default_value("default");
  field.UnsafeSetDefault(&default_value);
  EXPECT_EQ(std::string("default"), field.Get().to_string());
  field.Set(&default_value, WrapRawString("Test short"), NULL);
  EXPECT_EQ(std::string("Test short"), field.Get().to_string());
  field.Set(&default_value, WrapRawString("Test long long long long value"), NULL);
  EXPECT_EQ(std::string("Test long long long long value"), field.Get().to_string());
  field.Set(&default_value, std::RawString(), NULL);
  field.Destroy(&default_value, NULL);

  ArenaRawStringPtr field2;
  field2.UnsafeSetDefault(&default_value);
  std::RawString* mut = field2.Mutable(&default_value, NULL);
  EXPECT_EQ(mut, field2.Mutable(&default_value, NULL));
  EXPECT_EQ(mut, &field2.Get());
  EXPECT_NE(&default_value, mut);
  EXPECT_EQ(std::string("default"), mut->to_string());
  std::string longlongdata("Test long long long long value");
  mut->assign(longlongdata.c_str(), longlongdata.size());
  EXPECT_EQ(std::string("Test long long long long value"), field2.Get().to_string());
  field2.Destroy(&default_value, NULL);
}

TEST(ArenaRawStringPtrTest, ArenaRawStringPtrOnArena) {
  Arena arena;
  ArenaRawStringPtr field;
  std::RawString default_value("default");
  field.UnsafeSetDefault(&default_value);
  EXPECT_EQ(std::string("default"), field.Get().to_string());
  field.Set(&default_value, WrapRawString("Test short"), &arena);
  EXPECT_EQ(std::string("Test short"), field.Get().to_string());
  field.Set(&default_value, WrapRawString("Test long long long long value"),
            &arena);
  EXPECT_EQ(std::string("Test long long long long value"),field.Get().to_string());
  field.Set(&default_value, std::RawString(""), &arena);
  field.Destroy(&default_value, &arena);

  ArenaRawStringPtr field2;
  field2.UnsafeSetDefault(&default_value);
  std::RawString* mut = field2.Mutable(&default_value, &arena);
  EXPECT_EQ(mut, field2.Mutable(&default_value, &arena));
  EXPECT_EQ(mut, &field2.Get());
  EXPECT_NE(&default_value, mut);
  EXPECT_EQ(std::string("default"), mut->to_string());
  std::string longlongdata("Test long long long long value");
  mut->assign(longlongdata.c_str(), longlongdata.size());
  EXPECT_EQ(std::string("Test long long long long value"), field2.Get().to_string());
  field2.Destroy(&default_value, &arena);
}

TEST(ArenaRawStringPtrTest, ArenaRawStringPtrOnArenaNoSSO) {
  Arena arena;
  ArenaRawStringPtr field;
  std::RawString default_value("default");
  field.UnsafeSetDefault(&default_value);
  EXPECT_EQ(std::string("default"), field.Get().to_string());

  // Avoid triggering the SSO optimization by setting the string to something
  // larger than the internal buffer.
  field.Set(&default_value, WrapRawString("Test long long long long value"),
            &arena);
  EXPECT_EQ(std::string("Test long long long long value"),field.Get().to_string());
  field.Set(&default_value, std::string(""), &arena);
  field.Destroy(&default_value, &arena);

  ArenaRawStringPtr field2;
  field2.UnsafeSetDefault(&default_value);
  std::RawString* mut = field2.Mutable(&default_value, &arena);
  EXPECT_EQ(mut, field2.Mutable(&default_value, &arena));
  EXPECT_EQ(mut, &field2.Get());
  EXPECT_NE(&default_value, mut);
  EXPECT_EQ(std::string("default"), mut->to_string());
  std::string longlongdata("Test long long long long value");
  mut->assign(longlongdata.c_str(), longlongdata.size());
  EXPECT_EQ(std::string("Test long long long long value"), field2.Get().to_string());
  field2.Destroy(&default_value, &arena);
}


} // namespace protobuf
} // namespace google