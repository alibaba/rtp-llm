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

#include <memory>
#include <string>
#include <vector>

#include <google/protobuf/unittest_proto3_arena.pb.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/testing/googletest.h>
#include <gtest/gtest.h>

using proto3_arena_unittest::TestAllTypes;

namespace google {
namespace protobuf {
namespace {
// We selectively set/check a few representative fields rather than all fields
// as this test is only expected to cover the basics of arena support.
void SetAllFields(TestAllTypes* m) {
  m->set_optional_int32(100);
  m->set_optional_string("asdf");
  m->set_optional_bytes("jkl;");
  m->mutable_optional_nested_message()->set_bb(42);
  m->mutable_optional_foreign_message()->set_c(43);
  m->set_optional_nested_enum(
      proto3_arena_unittest::TestAllTypes_NestedEnum_BAZ);
  m->set_optional_foreign_enum(proto3_arena_unittest::FOREIGN_BAZ);
  m->mutable_optional_lazy_message()->set_bb(45);
  m->add_repeated_int32(100);
  m->add_repeated_string("asdf");
  m->add_repeated_bytes("jkl;");
  m->add_repeated_nested_message()->set_bb(46);
  m->add_repeated_foreign_message()->set_c(47);
  m->add_repeated_nested_enum(
      proto3_arena_unittest::TestAllTypes_NestedEnum_BAZ);
  m->add_repeated_foreign_enum(proto3_arena_unittest::FOREIGN_BAZ);
  m->add_repeated_lazy_message()->set_bb(49);

  m->set_oneof_uint32(1);
  m->mutable_oneof_nested_message()->set_bb(50);
  m->set_oneof_string("test");  // only this one remains set

  m->set_optional_rawstring("qwe");
  m->add_repeated_rawstring("asd");
}

void ExpectAllFieldsSet(const TestAllTypes& m) {
  EXPECT_EQ(100, m.optional_int32());
  EXPECT_EQ("asdf", m.optional_string());
  EXPECT_EQ("jkl;", m.optional_bytes());
  EXPECT_EQ(true, m.has_optional_nested_message());
  EXPECT_EQ(42, m.optional_nested_message().bb());
  EXPECT_EQ(true, m.has_optional_foreign_message());
  EXPECT_EQ(43, m.optional_foreign_message().c());
  EXPECT_EQ(proto3_arena_unittest::TestAllTypes_NestedEnum_BAZ,
            m.optional_nested_enum());
  EXPECT_EQ(proto3_arena_unittest::FOREIGN_BAZ, m.optional_foreign_enum());
  EXPECT_EQ(true, m.has_optional_lazy_message());
  EXPECT_EQ(45, m.optional_lazy_message().bb());

  EXPECT_EQ(1, m.repeated_int32_size());
  EXPECT_EQ(100, m.repeated_int32(0));
  EXPECT_EQ(1, m.repeated_string_size());
  EXPECT_EQ("asdf", m.repeated_string(0));
  EXPECT_EQ(1, m.repeated_bytes_size());
  EXPECT_EQ("jkl;", m.repeated_bytes(0));
  EXPECT_EQ(1, m.repeated_nested_message_size());
  EXPECT_EQ(46, m.repeated_nested_message(0).bb());
  EXPECT_EQ(1, m.repeated_foreign_message_size());
  EXPECT_EQ(47, m.repeated_foreign_message(0).c());
  EXPECT_EQ(1, m.repeated_nested_enum_size());
  EXPECT_EQ(proto3_arena_unittest::TestAllTypes_NestedEnum_BAZ,
            m.repeated_nested_enum(0));
  EXPECT_EQ(1, m.repeated_foreign_enum_size());
  EXPECT_EQ(proto3_arena_unittest::FOREIGN_BAZ, m.repeated_foreign_enum(0));
  EXPECT_EQ(1, m.repeated_lazy_message_size());
  EXPECT_EQ(49, m.repeated_lazy_message(0).bb());

  EXPECT_EQ(proto3_arena_unittest::TestAllTypes::kOneofString,
            m.oneof_field_case());
  EXPECT_EQ("test", m.oneof_string());

  EXPECT_EQ("qwe", m.optional_rawstring().to_string());
  EXPECT_EQ(1, m.repeated_rawstring_size());
  EXPECT_EQ("asd", m.repeated_rawstring(0).to_string());
}

// In this file we only test some basic functionalities of arena support in
// proto3 and expect the arena support to be fully tested in proto2 unittests
// because proto3 shares most code with proto2.

TEST(Proto3ArenaLiteTest, Parsing) {
  TestAllTypes original;
  SetAllFields(&original);

  Arena arena;
  TestAllTypes* arena_message = Arena::CreateMessage<TestAllTypes>(&arena);
  arena_message->ParseFromString(original.SerializeAsString());
  ExpectAllFieldsSet(*arena_message);
}

TEST(Proto3ArenaLiteTest, Swap) {
  Arena arena1;
  Arena arena2;

  // Test Swap().
  TestAllTypes* arena1_message = Arena::CreateMessage<TestAllTypes>(&arena1);
  TestAllTypes* arena2_message = Arena::CreateMessage<TestAllTypes>(&arena2);
  
  std::RawString rstr1("qwe");
  std::RawString rstr2("asd");
  arena1_message->set_optional_rawstring(rstr1);
  arena2_message->set_optional_rawstring(rstr2);

  arena1_message->Swap(arena2_message);
  EXPECT_EQ(&arena1, arena1_message->GetArena());
  EXPECT_EQ(&arena2, arena2_message->GetArena());
  EXPECT_EQ(rstr1.shared_data(), arena2_message->optional_rawstring().shared_data());
  EXPECT_EQ(rstr1.size(), arena2_message->optional_rawstring().size());
  EXPECT_EQ(rstr2.shared_data(), arena1_message->optional_rawstring().shared_data());
  EXPECT_EQ(rstr2.size(), arena1_message->optional_rawstring().size());
}

TEST(Proto3ArenaLiteTest, SetAllocatedMessage) {
  Arena arena;
  TestAllTypes* arena_message = Arena::CreateMessage<TestAllTypes>(&arena);
  TestAllTypes::NestedMessage* nested = new TestAllTypes::NestedMessage;
  nested->set_bb(118);
  arena_message->set_allocated_optional_nested_message(nested);
  EXPECT_EQ(118, arena_message->optional_nested_message().bb());
}

TEST(Proto3ArenaLiteTest, ReleaseMessage) {
  Arena arena;
  TestAllTypes* arena_message = Arena::CreateMessage<TestAllTypes>(&arena);
  arena_message->mutable_optional_nested_message()->set_bb(118);
  std::unique_ptr<TestAllTypes::NestedMessage> nested(
      arena_message->release_optional_nested_message());
  EXPECT_EQ(118, nested->bb());
}

}  // namespace
}  // namespace protobuf
}  // namespace google
