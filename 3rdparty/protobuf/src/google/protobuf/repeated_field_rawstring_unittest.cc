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

// Author: kenton@google.com (Kenton Varda)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.
//
// TODO(kenton):  Improve this unittest to bring it up to the standards of
//   other proto2 unittests.

#include <algorithm>
#include <limits>
#include <list>
#include <type_traits>
#include <vector>

#include <google/protobuf/repeated_field.h>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/unittest.pb.h>
#include <google/protobuf/stubs/strutil.h>
#include <gmock/gmock.h>
#include <google/protobuf/testing/googletest.h>
#include <gtest/gtest.h>

#include <google/protobuf/stubs/stl_util.h>

namespace google {
namespace protobuf {
namespace {

using ::protobuf_unittest::TestAllTypes;
using ::testing::ElementsAre;

// Test operations on a small RepeatedField.
TEST(RepeatedFieldOnRawString, Small) {
  RepeatedField<std::RawString> field;

  EXPECT_TRUE(field.empty());
  EXPECT_EQ(field.size(), 0);

  field.Add("5");

  EXPECT_FALSE(field.empty());
  EXPECT_EQ(field.size(), 1);
  EXPECT_EQ(field.Get(0).to_string(), "5");
  EXPECT_EQ(field.at(0).to_string(), "5");

  field.Add("42");

  EXPECT_FALSE(field.empty());
  EXPECT_EQ(field.size(), 2);
  EXPECT_EQ(field.Get(0).to_string(), "5");
  EXPECT_EQ(field.at(0).to_string(), "5");
  EXPECT_EQ(field.Get(1).to_string(), "42");
  EXPECT_EQ(field.at(1).to_string(), "42");

  field.Set(1, "23");

  EXPECT_FALSE(field.empty());
  EXPECT_EQ(field.size(), 2);
  EXPECT_EQ(field.Get(0).to_string(), "5");
  EXPECT_EQ(field.at(0).to_string(), "5");
  EXPECT_EQ(field.Get(1).to_string(), "23");
  EXPECT_EQ(field.at(1).to_string(), "23");

  field.at(1) = "25";

  EXPECT_FALSE(field.empty());
  EXPECT_EQ(field.size(), 2);
  EXPECT_EQ(field.Get(0).to_string(), "5");
  EXPECT_EQ(field.at(0).to_string(), "5");
  EXPECT_EQ(field.Get(1).to_string(), "25");
  EXPECT_EQ(field.at(1).to_string(), "25");

  field.RemoveLast();

  EXPECT_FALSE(field.empty());
  EXPECT_EQ(field.size(), 1);
  EXPECT_EQ(field.Get(0).to_string(), "5");
  EXPECT_EQ(field.at(0).to_string(), "5");

  field.Clear();

  EXPECT_TRUE(field.empty());
  EXPECT_EQ(field.size(), 0);
  // Additional bytes are for 'struct Rep' header.
  int expected_usage = 4 * sizeof(std::RawString) + sizeof(Arena*);
  EXPECT_GE(field.SpaceUsedExcludingSelf(), expected_usage);
}


// Test operations on a RepeatedField which is large enough to allocate a
// separate array.
TEST(RepeatedFieldOnRawString, Large) {
  RepeatedField<std::RawString> field;

  for (int i = 0; i < 16; i++) {
    field.Add(std::to_string(i * i));
  }

  EXPECT_FALSE(field.empty());
  EXPECT_EQ(field.size(), 16);

  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(field.Get(i).to_string(), std::to_string(i * i));
  }

  int expected_usage = 16 * sizeof(std::RawString);
  EXPECT_GE(field.SpaceUsedExcludingSelf(), expected_usage);
}

// Test swapping between various types of RepeatedFields.
TEST(RepeatedFieldOnRawString, SwapSmallSmall) {
  RepeatedField<std::RawString> field1;
  RepeatedField<std::RawString> field2;

  field1.Add("5");
  field1.Add("42");

  EXPECT_FALSE(field1.empty());
  EXPECT_EQ(field1.size(), 2);
  EXPECT_EQ(field1.Get(0).to_string(), "5");
  EXPECT_EQ(field1.Get(1).to_string(), "42");

  EXPECT_TRUE(field2.empty());
  EXPECT_EQ(field2.size(), 0);

  field1.Swap(&field2);

  EXPECT_TRUE(field1.empty());
  EXPECT_EQ(field1.size(), 0);

  EXPECT_FALSE(field2.empty());
  EXPECT_EQ(field2.size(), 2);
  EXPECT_EQ(field2.Get(0).to_string(), "5");
  EXPECT_EQ(field2.Get(1).to_string(), "42");
}

TEST(RepeatedFieldOnRawString, SwapLargeSmall) {
  RepeatedField<std::RawString> field1;
  RepeatedField<std::RawString> field2;

  for (int i = 0; i < 16; i++) {
    field1.Add(std::to_string(i * i));
  }
  field2.Add("5");
  field2.Add("42");
  field1.Swap(&field2);

  EXPECT_EQ(field1.size(), 2);
  EXPECT_EQ(field1.Get(0).to_string(), "5");
  EXPECT_EQ(field1.Get(1).to_string(), "42");
  EXPECT_EQ(field2.size(), 16);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(field2.Get(i).to_string(), std::to_string(i * i));
  }
}

TEST(RepeatedFieldOnRawString, SwapLargeLarge) {
  RepeatedField<std::RawString> field1;
  RepeatedField<std::RawString> field2;

  field1.Add("5");
  field1.Add("42");
  for (int i = 0; i < 16; i++) {
    field1.Add(std::to_string(i));
    field2.Add(std::to_string(i * i));
  }
  field2.Swap(&field1);

  EXPECT_EQ(field1.size(), 16);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(field1.Get(i).to_string(), std::to_string(i * i));
  }
  EXPECT_EQ(field2.size(), 18);
  EXPECT_EQ(field2.Get(0).to_string(), "5");
  EXPECT_EQ(field2.Get(1).to_string(), "42");
  for (int i = 2; i < 18; i++) {
    EXPECT_EQ(field2.Get(i).to_string(), std::to_string(i - 2));
  }
}

// Determines how much space was reserved by the given field by adding elements
// to it until it re-allocates its space.
static int ReservedSpace(RepeatedField<std::RawString>* field) {
  const std::RawString* ptr = field->data();
  do {
    field->Add("0");
  } while (field->data() == ptr);

  return field->size() - 1;
}

TEST(RepeatedFieldOnRawString, ReserveMoreThanDouble) {
  // Reserve more than double the previous space in the field and expect the
  // field to reserve exactly the amount specified.
  RepeatedField<std::RawString> field;
  field.Reserve(20);

  EXPECT_LE(20, ReservedSpace(&field));
}

TEST(RepeatedFieldOnRawString, ReserveLessThanDouble) {
  // Reserve less than double the previous space in the field and expect the
  // field to grow by double instead.
  RepeatedField<std::RawString> field;
  field.Reserve(20);
  int capacity = field.Capacity();
  field.Reserve(capacity * 1.5);

  EXPECT_LE(2 * capacity, ReservedSpace(&field));
}

TEST(RepeatedFieldOnRawString, ReserveLessThanExisting) {
  // Reserve less than the previous space in the field and expect the
  // field to not re-allocate at all.
  RepeatedField<std::RawString> field;
  field.Reserve(20);
  const std::RawString* previous_ptr = field.data();
  field.Reserve(10);

  EXPECT_EQ(previous_ptr, field.data());
  EXPECT_LE(20, ReservedSpace(&field));
}

TEST(RepeatedFieldOnRawString, Resize) {
  RepeatedField<std::RawString> field;
  field.Resize(2, "1");
  EXPECT_EQ(2, field.size());
  field.Resize(5, "2");
  EXPECT_EQ(5, field.size());
  field.Resize(4, "3");
  ASSERT_EQ(4, field.size());
  EXPECT_EQ("1", field.Get(0).to_string());
  EXPECT_EQ("1", field.Get(1).to_string());
  EXPECT_EQ("2", field.Get(2).to_string());
  EXPECT_EQ("2", field.Get(3).to_string());
  field.Resize(0, "4");
  EXPECT_TRUE(field.empty());
}

TEST(RepeatedFieldOnRawString, MergeFrom) {
  RepeatedField<std::RawString> source, destination;
  source.Add("4");
  source.Add("5");
  destination.Add("1");
  destination.Add("2");
  destination.Add("3");

  destination.MergeFrom(source);

  ASSERT_EQ(5, destination.size());
  EXPECT_EQ("1", destination.Get(0).to_string());
  EXPECT_EQ("2", destination.Get(1).to_string());
  EXPECT_EQ("3", destination.Get(2).to_string());
  EXPECT_EQ("4", destination.Get(3).to_string());
  EXPECT_EQ("5", destination.Get(4).to_string());
}


TEST(RepeatedFieldOnRawString, CopyFrom) {
  RepeatedField<std::RawString> source, destination;
  source.Add("4");
  source.Add("5");
  destination.Add("1");
  destination.Add("2");
  destination.Add("3");

  destination.CopyFrom(source);

  ASSERT_EQ(2, destination.size());
  EXPECT_EQ("4", destination.Get(0).to_string());
  EXPECT_EQ("5", destination.Get(1).to_string());
}

TEST(RepeatedFieldOnRawString, CopyFromSelf) {
  RepeatedField<std::RawString> me;
  me.Add("3");
  me.CopyFrom(me);
  ASSERT_EQ(1, me.size());
  EXPECT_EQ("3", me.Get(0).to_string());
}

TEST(RepeatedFieldOnRawString, Erase) {
  RepeatedField<std::RawString> me;
  RepeatedField<std::RawString>::iterator it = me.erase(me.begin(), me.end());
  EXPECT_TRUE(me.begin() == it);
  EXPECT_EQ(0, me.size());

  me.Add("1");
  me.Add("2");
  me.Add("3");
  it = me.erase(me.begin(), me.end());
  EXPECT_TRUE(me.begin() == it);
  EXPECT_EQ(0, me.size());

  me.Add("4");
  me.Add("5");
  me.Add("6");
  it = me.erase(me.begin() + 2, me.end());
  EXPECT_TRUE(me.begin() + 2 == it);
  EXPECT_EQ(2, me.size());
  EXPECT_EQ("4", me.Get(0).to_string());
  EXPECT_EQ("5", me.Get(1).to_string());

  me.Add("6");
  me.Add("7");
  me.Add("8");
  it = me.erase(me.begin() + 1, me.begin() + 3);
  EXPECT_TRUE(me.begin() + 1 == it);
  EXPECT_EQ(3, me.size());
  EXPECT_EQ("4", me.Get(0).to_string());
  EXPECT_EQ("7", me.Get(1).to_string());
  EXPECT_EQ("8", me.Get(2).to_string());
}

TEST(RepeatedFieldOnRawString, CopyConstruct) {
  RepeatedField<std::RawString> source;
  source.Add("1");
  source.Add("2");

  RepeatedField<std::RawString> destination(source);

  ASSERT_EQ(2, destination.size());
  EXPECT_EQ("1", destination.Get(0).to_string());
  EXPECT_EQ("2", destination.Get(1).to_string());
}

TEST(RepeatedFieldOnRawString, IteratorConstruct) {
  std::vector<std::RawString> values;
  RepeatedField<std::RawString> empty(values.begin(), values.end());
  ASSERT_EQ(values.size(), empty.size());

  values.push_back("1");
  values.push_back("2");

  RepeatedField<std::RawString> field(values.begin(), values.end());
  ASSERT_EQ(values.size(), field.size());
  EXPECT_EQ(values[0].size(), field.Get(0).size());
  EXPECT_EQ(values[1].size(), field.Get(1).size());

  RepeatedField<std::RawString> other(field.begin(), field.end());
  ASSERT_EQ(values.size(), other.size());
  EXPECT_EQ(values[0].size(), other.Get(0).size());
  EXPECT_EQ(values[1].size(), other.Get(1).size());
}

TEST(RepeatedFieldOnRawString, CopyAssign) {
  RepeatedField<std::RawString> source, destination;
  source.Add("4");
  source.Add("5");
  destination.Add("1");
  destination.Add("2");
  destination.Add("3");

  destination = source;

  ASSERT_EQ(2, destination.size());
  EXPECT_EQ("4", destination.Get(0).to_string());
  EXPECT_EQ("5", destination.Get(1).to_string());
}

TEST(RepeatedFieldOnRawString, SelfAssign) {
  // Verify that assignment to self does not destroy data.
  RepeatedField<std::RawString> source, *p;
  p = &source;
  source.Add("7");
  source.Add("8");

  *p = source;

  ASSERT_EQ(2, source.size());
  EXPECT_EQ("7", source.Get(0).to_string());
  EXPECT_EQ("8", source.Get(1).to_string());
}

TEST(RepeatedFieldOnRawString, MoveConstruct) {
  {
    RepeatedField<std::RawString> source;
    source.Add("1");
    source.Add("2");
    const std::RawString* data = source.data();
    RepeatedField<std::RawString> destination = std::move(source);
    EXPECT_EQ(data, destination.data());
    EXPECT_EQ("1", destination.Get(0).to_string());
    EXPECT_EQ("2", destination.Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_TRUE(source.empty());
  }
  {
    Arena arena;
    RepeatedField<std::RawString>* source =
        Arena::CreateMessage<RepeatedField<std::RawString>>(&arena);
    source->Add("1");
    source->Add("2");
    RepeatedField<std::RawString> destination = std::move(*source);
    EXPECT_EQ(nullptr, destination.GetArena());
    EXPECT_EQ("1", destination.Get(0).to_string());
    EXPECT_EQ("2", destination.Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_EQ("1", destination.Get(0).to_string());
    EXPECT_EQ("2", destination.Get(1).to_string());
  }
}

TEST(RepeatedFieldOnRawString, MoveAssign) {
  {
    RepeatedField<std::RawString> source;
    source.Add("1");
    source.Add("2");
    RepeatedField<std::RawString> destination;
    destination.Add("3");
    const std::RawString* source_data = source.data();
    const std::RawString* destination_data = destination.data();
    destination = std::move(source);
    EXPECT_EQ(source_data, destination.data());
    EXPECT_EQ("1", destination.Get(0).to_string());
    EXPECT_EQ("2", destination.Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_EQ(destination_data, source.data());
    EXPECT_EQ("3", source.Get(0).to_string());
  }
  {
    Arena arena;
    RepeatedField<std::RawString>* source =
        Arena::CreateMessage<RepeatedField<std::RawString>>(&arena);
    source->Add("1");
    source->Add("2");
    RepeatedField<std::RawString>* destination =
        Arena::CreateMessage<RepeatedField<std::RawString>>(&arena);
    destination->Add("3");
    const std::RawString* source_data = source->data();
    const std::RawString* destination_data = destination->data();
    *destination = std::move(*source);
    EXPECT_EQ(source_data, destination->data());
    EXPECT_EQ("1", destination->Get(0).to_string());
    EXPECT_EQ("2", destination->Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_EQ(destination_data, source->data());
    EXPECT_EQ("3", source->Get(0).to_string());
  }
  {
    Arena source_arena;
    RepeatedField<std::RawString>* source =
        Arena::CreateMessage<RepeatedField<std::RawString>>(&source_arena);
    source->Add("1");
    source->Add("2");
    Arena destination_arena;
    RepeatedField<std::RawString>* destination =
        Arena::CreateMessage<RepeatedField<std::RawString>>(&destination_arena);
    destination->Add("3");
    *destination = std::move(*source);
    EXPECT_EQ("1", destination->Get(0).to_string());
    EXPECT_EQ("2", destination->Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_EQ("1", source->Get(0).to_string());
    EXPECT_EQ("2", source->Get(1).to_string());
  }
  {
    Arena arena;
    RepeatedField<std::RawString>* source =
        Arena::CreateMessage<RepeatedField<std::RawString>>(&arena);
    source->Add("1");
    source->Add("2");
    RepeatedField<std::RawString> destination;
    destination.Add("3");
    destination = std::move(*source);
    EXPECT_EQ("1", destination.Get(0).to_string());
    EXPECT_EQ("2", destination.Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_EQ("1", source->Get(0).to_string());
    EXPECT_EQ("2", source->Get(1).to_string());
  }
  {
    RepeatedField<std::RawString> source;
    source.Add("1");
    source.Add("2");
    Arena arena;
    RepeatedField<std::RawString>* destination =
        Arena::CreateMessage<RepeatedField<std::RawString>>(&arena);
    destination->Add("3");
    *destination = std::move(source);
    EXPECT_EQ("1", destination->Get(0).to_string());
    EXPECT_EQ("2", destination->Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_EQ("1", source.Get(0).to_string());
    EXPECT_EQ("2", source.Get(1).to_string());
  }
  {
    RepeatedField<std::RawString> field;
    // An alias to defeat -Wself-move.
    RepeatedField<std::RawString>& alias = field;
    field.Add("1");
    field.Add("2");
    const std::RawString* data = field.data();
    field = std::move(alias);
    EXPECT_EQ(data, field.data());
    EXPECT_EQ("1", field.Get(0).to_string());
    EXPECT_EQ("2", field.Get(1).to_string());
  }
  {
    Arena arena;
    RepeatedField<std::RawString>* field =
        Arena::CreateMessage<RepeatedField<std::RawString>>(&arena);
    field->Add("1");
    field->Add("2");
    const std::RawString* data = field->data();
    *field = std::move(*field);
    EXPECT_EQ(data, field->data());
    EXPECT_EQ("1", field->Get(0).to_string());
    EXPECT_EQ("2", field->Get(1).to_string());
  }
}

TEST(Movable, Works) {
  class NonMoveConstructible {
   public:
    NonMoveConstructible(NonMoveConstructible&&) = delete;
    NonMoveConstructible& operator=(NonMoveConstructible&&) { return *this; }
  };
  class NonMoveAssignable {
   public:
    NonMoveAssignable(NonMoveAssignable&&) {}
    NonMoveAssignable& operator=(NonMoveConstructible&&) = delete;
  };
  class NonMovable {
   public:
    NonMovable(NonMovable&&) = delete;
    NonMovable& operator=(NonMovable&&) = delete;
  };

  EXPECT_TRUE(internal::IsMovable<std::RawString>::value);
  EXPECT_TRUE(internal::IsMovable<std::RawString>::value);

  EXPECT_FALSE(std::is_move_constructible<NonMoveConstructible>::value);
  EXPECT_TRUE(std::is_move_assignable<NonMoveConstructible>::value);
  EXPECT_FALSE(internal::IsMovable<NonMoveConstructible>::value);

  EXPECT_TRUE(std::is_move_constructible<NonMoveAssignable>::value);
  EXPECT_FALSE(std::is_move_assignable<NonMoveAssignable>::value);
  EXPECT_FALSE(internal::IsMovable<NonMoveAssignable>::value);

  EXPECT_FALSE(internal::IsMovable<NonMovable>::value);
}

TEST(RepeatedFieldOnRawString, MutableDataIsMutable) {
  RepeatedField<std::RawString> field;
  field.Add("1");
  EXPECT_EQ("1", field.Get(0).to_string());
  // The fact that this line compiles would be enough, but we'll check the
  // value anyway.
  *field.mutable_data() = "2";
  EXPECT_EQ("2", field.Get(0).to_string());
}

TEST(RepeatedFieldOnRawString, SubscriptOperators) {
  RepeatedField<std::RawString> field;
  field.Add("1");
  EXPECT_EQ("1", field.Get(0).to_string());
  EXPECT_EQ("1", field[0].to_string());
  EXPECT_EQ(field.Mutable(0), &field[0]);
  const RepeatedField<std::RawString>& const_field = field;
  EXPECT_EQ(field.data(), &const_field[0]);
}

TEST(RepeatedFieldOnRawString, Truncate) {
  RepeatedField<std::RawString> field;

  field.Add("12");
  field.Add("34");
  field.Add("56");
  field.Add("78");
  EXPECT_EQ(4, field.size());

  field.Truncate(3);
  EXPECT_EQ(3, field.size());

  field.Add("90");
  EXPECT_EQ(4, field.size());
  EXPECT_EQ("90", field.Get(3).to_string());

  // Truncations that don't change the size are allowed, but growing is not
  // allowed.
  field.Truncate(field.size());
#ifdef PROTOBUF_HAS_DEATH_TEST
  EXPECT_DEBUG_DEATH(field.Truncate(field.size() + 1), "new_size");
#endif
}

TEST(RepeatedFieldOnRawString, ClearThenReserveMore) {
  // Test that Reserve properly destroys the old internal array when it's forced
  // to allocate a new one, even when cleared-but-not-deleted objects are
  // present. Use a 'string' and > 16 bytes length so that the elements are
  // non-POD and allocate -- the leak checker will catch any skipped destructor
  // calls here.
  RepeatedField<std::RawString> field;
  for (int i = 0; i < 32; i++) {
    field.Add(std::RawString("abcdefghijklmnopqrstuvwxyz0123456789"));
  }
  EXPECT_EQ(32, field.size());
  field.Clear();
  EXPECT_EQ(0, field.size());
  EXPECT_LE(32, field.Capacity());

  field.Reserve(1024);
  EXPECT_EQ(0, field.size());
  EXPECT_LE(1024, field.Capacity());
  // Finish test -- |field| should destroy the cleared-but-not-yet-destroyed
  // strings.
}

// ===================================================================
// RepeatedPtrField tests.  These pretty much just mirror the RepeatedField
// tests above.

TEST(RepeatedPtrFieldOnRawString, Small) {
  RepeatedPtrField<std::RawString> field;

  EXPECT_TRUE(field.empty());
  EXPECT_EQ(field.size(), 0);

  field.Add()->assign("foo");

  EXPECT_FALSE(field.empty());
  EXPECT_EQ(field.size(), 1);
  EXPECT_EQ(field.Get(0).to_string(), "foo");
  EXPECT_EQ(field.at(0).to_string(), "foo");

  field.Add()->assign("bar");

  EXPECT_FALSE(field.empty());
  EXPECT_EQ(field.size(), 2);
  EXPECT_EQ(field.Get(0).to_string(), "foo");
  EXPECT_EQ(field.at(0).to_string(), "foo");
  EXPECT_EQ(field.Get(1).to_string(), "bar");
  EXPECT_EQ(field.at(1).to_string(), "bar");

  field.Mutable(1)->assign("baz");

  EXPECT_FALSE(field.empty());
  EXPECT_EQ(field.size(), 2);
  EXPECT_EQ(field.Get(0).to_string(), "foo");
  EXPECT_EQ(field.at(0).to_string(), "foo");
  EXPECT_EQ(field.Get(1).to_string(), "baz");
  EXPECT_EQ(field.at(1).to_string(), "baz");

  field.RemoveLast();

  EXPECT_FALSE(field.empty());
  EXPECT_EQ(field.size(), 1);
  EXPECT_EQ(field.Get(0).to_string(), "foo");
  EXPECT_EQ(field.at(0).to_string(), "foo");

  field.Clear();

  EXPECT_TRUE(field.empty());
  EXPECT_EQ(field.size(), 0);
}

TEST(RepeatedPtrFieldOnRawString, Large) {
  RepeatedPtrField<std::RawString> field;

  for (int i = 0; i < 16; i++) {
    field.Add()->assign(std::to_string(i));
  }

  EXPECT_EQ(field.size(), 16);

  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(field.Get(i).to_string(), std::to_string(i));
  }

  int min_expected_usage = 16 * sizeof(std::RawString);
  EXPECT_GE(field.SpaceUsedExcludingSelf(), min_expected_usage);
}

TEST(RepeatedPtrFieldOnRawString, SwapSmallSmall) {
  RepeatedPtrField<std::RawString> field1;
  RepeatedPtrField<std::RawString> field2;

  EXPECT_TRUE(field1.empty());
  EXPECT_EQ(field1.size(), 0);
  EXPECT_TRUE(field2.empty());
  EXPECT_EQ(field2.size(), 0);

  field1.Add()->assign("foo");
  field1.Add()->assign("bar");

  EXPECT_FALSE(field1.empty());
  EXPECT_EQ(field1.size(), 2);
  EXPECT_EQ(field1.Get(0).to_string(), "foo");
  EXPECT_EQ(field1.Get(1).to_string(), "bar");

  EXPECT_TRUE(field2.empty());
  EXPECT_EQ(field2.size(), 0);

  field1.Swap(&field2);

  EXPECT_TRUE(field1.empty());
  EXPECT_EQ(field1.size(), 0);

  EXPECT_EQ(field2.size(), 2);
  EXPECT_EQ(field2.Get(0).to_string(), "foo");
  EXPECT_EQ(field2.Get(1).to_string(), "bar");
}

TEST(RepeatedPtrFieldOnRawString, SwapLargeSmall) {
  RepeatedPtrField<std::RawString> field1;
  RepeatedPtrField<std::RawString> field2;

  field2.Add()->assign("foo");
  field2.Add()->assign("bar");
  for (int i = 0; i < 16; i++) {
    std::string value;
    value += 'a' + i;
    field1.Add()->assign(value);
  }
  field1.Swap(&field2);

  EXPECT_EQ(field1.size(), 2);
  EXPECT_EQ(field1.Get(0).to_string(), "foo");
  EXPECT_EQ(field1.Get(1).to_string(), "bar");
  EXPECT_EQ(field2.size(), 16);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(field2.Get(i).size(), 1);
    EXPECT_EQ(field2.Get(i).data()[0], 'a' + i);
  }
}

TEST(RepeatedPtrFieldOnRawString, SwapLargeLarge) {
  RepeatedPtrField<std::RawString> field1;
  RepeatedPtrField<std::RawString> field2;

  field1.Add()->assign("foo");
  field1.Add()->assign("bar");
  for (int i = 0; i < 16; i++) {
    std::string value1;
    value1 += 'A' + i;
    field1.Add()->assign(value1);
    std::string value2;
    value2 += 'a' + i;
    field2.Add()->assign(value2);
  }
  field2.Swap(&field1);

  EXPECT_EQ(field1.size(), 16);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(field1.Get(i).size(), 1);
    EXPECT_EQ(field1.Get(i).data()[0], 'a' + i);
  }
  EXPECT_EQ(field2.size(), 18);
  EXPECT_EQ(field2.Get(0).to_string(), "foo");
  EXPECT_EQ(field2.Get(1).to_string(), "bar");
  for (int i = 2; i < 18; i++) {
    EXPECT_EQ(field2.Get(i).size(), 1);
    EXPECT_EQ(field2.Get(i).data()[0], 'A' + i - 2);
  }
}

static int ReservedSpace(RepeatedPtrField<std::RawString>* field) {
  const std::RawString* const* ptr = field->data();
  do {
    field->Add();
  } while (field->data() == ptr);

  return field->size() - 1;
}

TEST(RepeatedPtrFieldOnRawString, ReserveMoreThanDouble) {
  RepeatedPtrField<std::RawString> field;
  field.Reserve(20);

  EXPECT_LE(20, ReservedSpace(&field));
}

TEST(RepeatedPtrFieldOnRawString, ReserveLessThanDouble) {
  RepeatedPtrField<std::RawString> field;
  field.Reserve(20);

  int capacity = field.Capacity();
  // Grow by 1.5x
  field.Reserve(capacity + (capacity >> 2));

  EXPECT_LE(2 * capacity, ReservedSpace(&field));
}

TEST(RepeatedPtrFieldOnRawString, ReserveLessThanExisting) {
  RepeatedPtrField<std::RawString> field;
  field.Reserve(20);
  const std::RawString* const* previous_ptr = field.data();
  field.Reserve(10);

  EXPECT_EQ(previous_ptr, field.data());
  EXPECT_LE(20, ReservedSpace(&field));
}

TEST(RepeatedPtrFieldOnRawString, ReserveDoesntLoseAllocated) {
  // Check that a bug is fixed:  An earlier implementation of Reserve()
  // failed to copy pointers to allocated-but-cleared objects, possibly
  // leading to segfaults.
  RepeatedPtrField<std::RawString> field;
  std::RawString* first = field.Add();
  field.RemoveLast();

  field.Reserve(20);
  EXPECT_EQ(first, field.Add());
}

// Clearing elements is tricky with RepeatedPtrFields since the memory for
// the elements is retained and reused.
TEST(RepeatedPtrFieldOnRawString, ClearedElements) {
  RepeatedPtrField<std::RawString> field;

  std::RawString* original = field.Add();
  *original = "foo";

  EXPECT_EQ(field.ClearedCount(), 0);

  field.RemoveLast();
  EXPECT_TRUE(original->size() == 0);
  EXPECT_EQ(field.ClearedCount(), 1);

  EXPECT_EQ(field.Add(),
            original);  // Should return same string for reuse.

  EXPECT_EQ(field.ReleaseLast(), original);  // We take ownership.
  EXPECT_EQ(field.ClearedCount(), 0);

  EXPECT_NE(field.Add(), original);  // Should NOT return the same string.
  EXPECT_EQ(field.ClearedCount(), 0);

  field.AddAllocated(original);  // Give ownership back.
  EXPECT_EQ(field.ClearedCount(), 0);
  EXPECT_EQ(field.Mutable(1), original);

  field.Clear();
  EXPECT_EQ(field.ClearedCount(), 2);
  EXPECT_EQ(field.ReleaseCleared(), original);  // Take ownership again.
  EXPECT_EQ(field.ClearedCount(), 1);
  EXPECT_NE(field.Add(), original);
  EXPECT_EQ(field.ClearedCount(), 0);
  EXPECT_NE(field.Add(), original);
  EXPECT_EQ(field.ClearedCount(), 0);

  field.AddCleared(original);  // Give ownership back, but as a cleared object.
  EXPECT_EQ(field.ClearedCount(), 1);
  EXPECT_EQ(field.Add(), original);
  EXPECT_EQ(field.ClearedCount(), 0);
}

// Test all code paths in AddAllocated().
TEST(RepeatedPtrFieldOnRawString, AddAlocated) {
  RepeatedPtrField<std::RawString> field;
  while (field.size() < field.Capacity()) {
    field.Add()->assign("filler");
  }

  int index = field.size();

  // First branch:  Field is at capacity with no cleared objects.
  std::RawString* foo = new std::RawString("foo");
  field.AddAllocated(foo);
  EXPECT_EQ(index + 1, field.size());
  EXPECT_EQ(0, field.ClearedCount());
  EXPECT_EQ(foo, &field.Get(index));

  // Last branch:  Field is not at capacity and there are no cleared objects.
  std::RawString* bar = new std::RawString("bar");
  field.AddAllocated(bar);
  ++index;
  EXPECT_EQ(index + 1, field.size());
  EXPECT_EQ(0, field.ClearedCount());
  EXPECT_EQ(bar, &field.Get(index));

  // Third branch:  Field is not at capacity and there are no cleared objects.
  field.RemoveLast();
  std::RawString* baz = new std::RawString("baz");
  field.AddAllocated(baz);
  EXPECT_EQ(index + 1, field.size());
  EXPECT_EQ(1, field.ClearedCount());
  EXPECT_EQ(baz, &field.Get(index));

  // Second branch:  Field is at capacity but has some cleared objects.
  while (field.size() < field.Capacity()) {
    field.Add()->assign("filler2");
  }
  field.RemoveLast();
  index = field.size();
  std::RawString* qux = new std::RawString("qux");
  field.AddAllocated(qux);
  EXPECT_EQ(index + 1, field.size());
  // We should have discarded the cleared object.
  EXPECT_EQ(0, field.ClearedCount());
  EXPECT_EQ(qux, &field.Get(index));
}

TEST(RepeatedPtrFieldOnRawString, MergeFrom) {
  RepeatedPtrField<std::RawString> source, destination;
  source.Add()->assign("4");
  source.Add()->assign("5");
  destination.Add()->assign("1");
  destination.Add()->assign("2");
  destination.Add()->assign("3");

  destination.MergeFrom(source);

  ASSERT_EQ(5, destination.size());
  EXPECT_EQ("1", destination.Get(0).to_string());
  EXPECT_EQ("2", destination.Get(1).to_string());
  EXPECT_EQ("3", destination.Get(2).to_string());
  EXPECT_EQ("4", destination.Get(3).to_string());
  EXPECT_EQ("5", destination.Get(4).to_string());
}


TEST(RepeatedPtrFieldOnRawString, CopyFrom) {
  RepeatedPtrField<std::RawString> source, destination;
  source.Add()->assign("4");
  source.Add()->assign("5");
  destination.Add()->assign("1");
  destination.Add()->assign("2");
  destination.Add()->assign("3");

  destination.CopyFrom(source);

  ASSERT_EQ(2, destination.size());
  EXPECT_EQ("4", destination.Get(0).to_string());
  EXPECT_EQ("5", destination.Get(1).to_string());
}

TEST(RepeatedPtrFieldOnRawString, CopyFromSelf) {
  RepeatedPtrField<std::RawString> me;
  me.Add()->assign("1");
  me.CopyFrom(me);
  ASSERT_EQ(1, me.size());
  EXPECT_EQ("1", me.Get(0).to_string());
}

TEST(RepeatedPtrFieldOnRawString, Erase) {
  RepeatedPtrField<std::RawString> me;
  RepeatedPtrField<std::RawString>::iterator it = me.erase(me.begin(), me.end());
  EXPECT_TRUE(me.begin() == it);
  EXPECT_EQ(0, me.size());

  *me.Add() = "1";
  *me.Add() = "2";
  *me.Add() = "3";
  it = me.erase(me.begin(), me.end());
  EXPECT_TRUE(me.begin() == it);
  EXPECT_EQ(0, me.size());

  *me.Add() = "4";
  *me.Add() = "5";
  *me.Add() = "6";
  it = me.erase(me.begin() + 2, me.end());
  EXPECT_TRUE(me.begin() + 2 == it);
  EXPECT_EQ(2, me.size());
  EXPECT_EQ("4", me.Get(0).to_string());
  EXPECT_EQ("5", me.Get(1).to_string());

  *me.Add() = "6";
  *me.Add() = "7";
  *me.Add() = "8";
  it = me.erase(me.begin() + 1, me.begin() + 3);
  EXPECT_TRUE(me.begin() + 1 == it);
  EXPECT_EQ(3, me.size());
  EXPECT_EQ("4", me.Get(0).to_string());
  EXPECT_EQ("7", me.Get(1).to_string());
  EXPECT_EQ("8", me.Get(2).to_string());
}

TEST(RepeatedPtrFieldOnRawString, CopyConstruct) {
  RepeatedPtrField<std::RawString> source;
  source.Add()->assign("1");
  source.Add()->assign("2");

  RepeatedPtrField<std::RawString> destination(source);

  ASSERT_EQ(2, destination.size());
  EXPECT_EQ("1", destination.Get(0).to_string());
  EXPECT_EQ("2", destination.Get(1).to_string());
}

TEST(RepeatedPtrFieldOnRawString, IteratorConstruct_String) {
  std::vector<std::string> values;
  values.push_back("1");
  values.push_back("2");

  RepeatedPtrField<std::RawString> field(values.begin(), values.end());
  ASSERT_EQ(values.size(), field.size());
  EXPECT_EQ(values[0], field.Get(0).to_string());
  EXPECT_EQ(values[1], field.Get(1).to_string());

  RepeatedPtrField<std::RawString> other(field.begin(), field.end());
  ASSERT_EQ(values.size(), other.size());
  EXPECT_EQ(values[0], other.Get(0).to_string());
  EXPECT_EQ(values[1], other.Get(1).to_string());
}

TEST(RepeatedPtrFieldOnRawString, CopyAssign) {
  RepeatedPtrField<std::RawString> source, destination;
  source.Add()->assign("4");
  source.Add()->assign("5");
  destination.Add()->assign("1");
  destination.Add()->assign("2");
  destination.Add()->assign("3");

  destination = source;

  ASSERT_EQ(2, destination.size());
  EXPECT_EQ("4", destination.Get(0).to_string());
  EXPECT_EQ("5", destination.Get(1).to_string());
}

TEST(RepeatedPtrFieldOnRawString, SelfAssign) {
  // Verify that assignment to self does not destroy data.
  RepeatedPtrField<std::RawString> source, *p;
  p = &source;
  source.Add()->assign("7");
  source.Add()->assign("8");

  *p = source;

  ASSERT_EQ(2, source.size());
  EXPECT_EQ("7", source.Get(0).to_string());
  EXPECT_EQ("8", source.Get(1).to_string());
}

TEST(RepeatedPtrFieldOnRawString, MoveConstruct) {
  {
    RepeatedPtrField<std::RawString> source;
    *source.Add() = "1";
    *source.Add() = "2";
    const std::RawString* const* data = source.data();
    RepeatedPtrField<std::RawString> destination = std::move(source);
    EXPECT_EQ(data, destination.data());
    EXPECT_EQ("1", destination.Get(0).to_string());
    EXPECT_EQ("2", destination.Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_TRUE(source.empty());
  }
  {
    Arena arena;
    RepeatedPtrField<std::RawString>* source =
        Arena::CreateMessage<RepeatedPtrField<std::RawString>>(&arena);
    *source->Add() = "1";
    *source->Add() = "2";
    RepeatedPtrField<std::RawString> destination = std::move(*source);
    EXPECT_EQ(nullptr, destination.GetArena());
    EXPECT_EQ("1", destination.Get(0).to_string());
    EXPECT_EQ("2", destination.Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_EQ("1", source->Get(0).to_string());
    EXPECT_EQ("2", source->Get(1).to_string());
  }
}

TEST(RepeatedPtrFieldOnRawString, MoveAssign) {
  {
    RepeatedPtrField<std::RawString> source;
    *source.Add() = "1";
    *source.Add() = "2";
    RepeatedPtrField<std::RawString> destination;
    *destination.Add() = "3";
    const std::RawString* const* source_data = source.data();
    const std::RawString* const* destination_data = destination.data();
    destination = std::move(source);
    EXPECT_EQ(source_data, destination.data());
    EXPECT_EQ("1", destination.Get(0).to_string());
    EXPECT_EQ("2", destination.Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_EQ(destination_data, source.data());
    EXPECT_EQ("3", source.Get(0).to_string());
  }
  {
    Arena arena;
    RepeatedPtrField<std::RawString>* source =
        Arena::CreateMessage<RepeatedPtrField<std::RawString>>(&arena);
    *source->Add() = "1";
    *source->Add() = "2";
    RepeatedPtrField<std::RawString>* destination =
        Arena::CreateMessage<RepeatedPtrField<std::RawString>>(&arena);
    *destination->Add() = "3";
    const std::RawString* const* source_data = source->data();
    const std::RawString* const* destination_data = destination->data();
    *destination = std::move(*source);
    EXPECT_EQ(source_data, destination->data());
    EXPECT_EQ("1", destination->Get(0).to_string());
    EXPECT_EQ("2", destination->Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_EQ(destination_data, source->data());
    EXPECT_THAT(*source, ElementsAre("3"));
    EXPECT_EQ("3", source->Get(0).to_string());
  }
  {
    Arena source_arena;
    RepeatedPtrField<std::RawString>* source =
        Arena::CreateMessage<RepeatedPtrField<std::RawString>>(&source_arena);
    *source->Add() = "1";
    *source->Add() = "2";
    Arena destination_arena;
    RepeatedPtrField<std::RawString>* destination =
        Arena::CreateMessage<RepeatedPtrField<std::RawString>>(&destination_arena);
    *destination->Add() = "3";
    *destination = std::move(*source);
    EXPECT_EQ("1", destination->Get(0).to_string());
    EXPECT_EQ("2", destination->Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_EQ("1", source->Get(0).to_string());
    EXPECT_EQ("2", source->Get(1).to_string());
  }
  {
    Arena arena;
    RepeatedPtrField<std::RawString>* source =
        Arena::CreateMessage<RepeatedPtrField<std::RawString>>(&arena);
    *source->Add() = "1";
    *source->Add() = "2";
    RepeatedPtrField<std::RawString> destination;
    *destination.Add() = "3";
    destination = std::move(*source);
    EXPECT_EQ("1", destination.Get(0).to_string());
    EXPECT_EQ("2", destination.Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_EQ("1", source->Get(0).to_string());
    EXPECT_EQ("2", source->Get(1).to_string());
  }
  {
    RepeatedPtrField<std::RawString> source;
    *source.Add() = "1";
    *source.Add() = "2";
    Arena arena;
    RepeatedPtrField<std::RawString>* destination =
        Arena::CreateMessage<RepeatedPtrField<std::RawString>>(&arena);
    *destination->Add() = "3";
    *destination = std::move(source);
    EXPECT_EQ("1", destination->Get(0).to_string());
    EXPECT_EQ("2", destination->Get(1).to_string());
    // This property isn't guaranteed but it's useful to have a test that would
    // catch changes in this area.
    EXPECT_EQ("1", source.Get(0).to_string());
    EXPECT_EQ("2", source.Get(1).to_string());
  }
  {
    RepeatedPtrField<std::RawString> field;
    // An alias to defeat -Wself-move.
    RepeatedPtrField<std::RawString>& alias = field;
    *field.Add() = "1";
    *field.Add() = "2";
    const std::RawString* const* data = field.data();
    field = std::move(alias);
    EXPECT_EQ(data, field.data());
    EXPECT_EQ("1", field.Get(0).to_string());
    EXPECT_EQ("2", field.Get(1).to_string());
  }
  {
    Arena arena;
    RepeatedPtrField<std::RawString>* field =
        Arena::CreateMessage<RepeatedPtrField<std::RawString>>(&arena);
    *field->Add() = "1";
    *field->Add() = "2";
    const std::RawString* const* data = field->data();
    *field = std::move(*field);
    EXPECT_EQ(data, field->data());
    EXPECT_EQ("1", field->Get(0).to_string());
    EXPECT_EQ("2", field->Get(1).to_string());
  }
}

TEST(RepeatedPtrFieldOnRawString, MutableDataIsMutable) {
  RepeatedPtrField<std::RawString> field;
  *field.Add() = "1";
  EXPECT_EQ("1", field.Get(0).to_string());
  // The fact that this line compiles would be enough, but we'll check the
  // value anyway.
  std::RawString** data = field.mutable_data();
  **data = "2";
  EXPECT_EQ("2", field.Get(0).to_string());
}

TEST(RepeatedPtrFieldOnRawString, SubscriptOperators) {
  RepeatedPtrField<std::RawString> field;
  *field.Add() = "1";
  EXPECT_EQ("1", field.Get(0).to_string());
  EXPECT_EQ("1", field[0].to_string());
  EXPECT_EQ(field.Mutable(0), &field[0]);
  const RepeatedPtrField<std::RawString>& const_field = field;
  EXPECT_EQ(*field.data(), &const_field[0]);
}

// ===================================================================

// Iterator tests stolen from net/proto/proto-array_unittest.
class RepeatedFieldIteratorTestOnRawString : public testing::Test {
 protected:
  virtual void SetUp() {
    for (int i = 0; i < 3; ++i) {
      proto_array_.Add(std::to_string(i));
    }
  }

  RepeatedField<std::RawString> proto_array_;
};

TEST_F(RepeatedFieldIteratorTestOnRawString, Convertible) {
  RepeatedField<std::RawString>::iterator iter = proto_array_.begin();
  RepeatedField<std::RawString>::const_iterator c_iter = iter;
  RepeatedField<std::RawString>::value_type value = *c_iter;
  EXPECT_EQ("0", value.to_string());
}

TEST_F(RepeatedFieldIteratorTestOnRawString, MutableIteration) {
  RepeatedField<std::RawString>::iterator iter = proto_array_.begin();
  EXPECT_EQ("0", (*iter).to_string());
  ++iter;
  EXPECT_EQ("1", (*iter).to_string());
  iter++;
  EXPECT_EQ("2", (*iter).to_string());
  ++iter;
  EXPECT_TRUE(proto_array_.end() == iter);

  EXPECT_EQ("2", (*(proto_array_.end() - 1)).to_string());
}

TEST_F(RepeatedFieldIteratorTestOnRawString, ConstIteration) {
  const RepeatedField<std::RawString>& const_proto_array = proto_array_;
  RepeatedField<std::RawString>::const_iterator iter = const_proto_array.begin();
  EXPECT_EQ("0", (*iter).to_string());
  ++iter;
  EXPECT_EQ("1", (*iter).to_string());
  iter++;
  EXPECT_EQ("2", (*iter).to_string());
  ++iter;
  EXPECT_TRUE(proto_array_.end() == iter);
  EXPECT_EQ("2", (*(proto_array_.end() - 1)).to_string());
}

TEST_F(RepeatedFieldIteratorTestOnRawString, Mutation) {
  RepeatedField<std::RawString>::iterator iter = proto_array_.begin();
  *iter = "7";
  EXPECT_EQ("7", proto_array_.Get(0).to_string());
}

// -------------------------------------------------------------------

class RepeatedPtrFieldIteratorTestOnRawString : public testing::Test {
 protected:
  virtual void SetUp() {
    proto_array_.Add()->assign("foo");
    proto_array_.Add()->assign("bar");
    proto_array_.Add()->assign("baz");
  }

  RepeatedPtrField<std::RawString> proto_array_;
};

TEST_F(RepeatedPtrFieldIteratorTestOnRawString, Convertible) {
  RepeatedPtrField<std::RawString>::iterator iter = proto_array_.begin();
  RepeatedPtrField<std::RawString>::const_iterator c_iter = iter;
  RepeatedPtrField<std::RawString>::value_type value = *c_iter;
  EXPECT_EQ("foo", value.to_string());
}

TEST_F(RepeatedPtrFieldIteratorTestOnRawString, MutableIteration) {
  RepeatedPtrField<std::RawString>::iterator iter = proto_array_.begin();
  EXPECT_EQ("foo", (*iter).to_string());
  ++iter;
  EXPECT_EQ("bar", (*(iter)).to_string());
  iter++;
  EXPECT_EQ("baz", (*iter).to_string());
  ++iter;
  EXPECT_TRUE(proto_array_.end() == iter);
  EXPECT_EQ("baz", (*(--proto_array_.end())).to_string());
}

TEST_F(RepeatedPtrFieldIteratorTestOnRawString, ConstIteration) {
  const RepeatedPtrField<std::RawString>& const_proto_array = proto_array_;
  RepeatedPtrField<std::RawString>::const_iterator iter =
      const_proto_array.begin();
  EXPECT_EQ("foo", (*iter).to_string());
  ++iter;
  EXPECT_EQ("bar", (*(iter)).to_string());
  iter++;
  EXPECT_EQ("baz", (*iter).to_string());
  ++iter;
  EXPECT_TRUE(const_proto_array.end() == iter);
  EXPECT_EQ("baz", (*(--const_proto_array.end())).to_string());
}

TEST_F(RepeatedPtrFieldIteratorTestOnRawString, MutableReverseIteration) {
  RepeatedPtrField<std::RawString>::reverse_iterator iter = proto_array_.rbegin();
  EXPECT_EQ("baz", (*iter).to_string());
  ++iter;
  EXPECT_EQ("bar", (*(iter)).to_string());
  iter++;
  EXPECT_EQ("foo", (*iter).to_string());
  ++iter;
  EXPECT_TRUE(proto_array_.rend() == iter);
  EXPECT_EQ("foo", (*(--proto_array_.rend())).to_string());
}

TEST_F(RepeatedPtrFieldIteratorTestOnRawString, ConstReverseIteration) {
  const RepeatedPtrField<std::RawString>& const_proto_array = proto_array_;
  RepeatedPtrField<std::RawString>::const_reverse_iterator iter =
      const_proto_array.rbegin();
  EXPECT_EQ("baz", (*iter).to_string());
  ++iter;
  EXPECT_EQ("bar", (*(iter)).to_string());
  iter++;
  EXPECT_EQ("foo", (*iter).to_string());
  ++iter;
  EXPECT_TRUE(const_proto_array.rend() == iter);
  EXPECT_EQ("foo", (*(--const_proto_array.rend())).to_string());
}

TEST_F(RepeatedPtrFieldIteratorTestOnRawString, RandomAccess) {
  RepeatedPtrField<std::RawString>::iterator iter = proto_array_.begin();
  RepeatedPtrField<std::RawString>::iterator iter2 = iter;
  ++iter2;
  ++iter2;
  EXPECT_TRUE(iter + 2 == iter2);
  EXPECT_TRUE(iter == iter2 - 2);
  EXPECT_EQ("baz", iter[2].to_string());
  EXPECT_EQ("baz", (*(iter + 2)).to_string());
  EXPECT_EQ(3, proto_array_.end() - proto_array_.begin());
}

TEST_F(RepeatedPtrFieldIteratorTestOnRawString, Comparable) {
  RepeatedPtrField<std::RawString>::const_iterator iter = proto_array_.begin();
  RepeatedPtrField<std::RawString>::const_iterator iter2 = iter + 1;
  EXPECT_TRUE(iter == iter);
  EXPECT_TRUE(iter != iter2);
  EXPECT_TRUE(iter < iter2);
  EXPECT_TRUE(iter <= iter2);
  EXPECT_TRUE(iter <= iter);
  EXPECT_TRUE(iter2 > iter);
  EXPECT_TRUE(iter2 >= iter);
  EXPECT_TRUE(iter >= iter);
}

// Uninitialized iterator does not point to any of the RepeatedPtrField.
TEST_F(RepeatedPtrFieldIteratorTestOnRawString, UninitializedIterator) {
  RepeatedPtrField<std::RawString>::iterator iter;
  EXPECT_TRUE(iter != proto_array_.begin());
  EXPECT_TRUE(iter != proto_array_.begin() + 1);
  EXPECT_TRUE(iter != proto_array_.begin() + 2);
  EXPECT_TRUE(iter != proto_array_.begin() + 3);
  EXPECT_TRUE(iter != proto_array_.end());
}

TEST_F(RepeatedPtrFieldIteratorTestOnRawString, Mutation) {
  RepeatedPtrField<std::RawString>::iterator iter = proto_array_.begin();
  *iter = "qux";
  EXPECT_EQ("qux", proto_array_.Get(0).to_string());
}

// -------------------------------------------------------------------

class RepeatedPtrFieldPtrsIteratorTestOnRawString : public testing::Test {
 protected:
  virtual void SetUp() {
    proto_array_.Add()->assign("foo");
    proto_array_.Add()->assign("bar");
    proto_array_.Add()->assign("baz");
    const_proto_array_ = &proto_array_;
  }

  RepeatedPtrField<std::RawString> proto_array_;
  const RepeatedPtrField<std::RawString>* const_proto_array_;
};

TEST_F(RepeatedPtrFieldPtrsIteratorTestOnRawString, ConvertiblePtr) {
  RepeatedPtrField<std::RawString>::pointer_iterator iter =
      proto_array_.pointer_begin();
  static_cast<void>(iter);
}

TEST_F(RepeatedPtrFieldPtrsIteratorTestOnRawString, ConvertibleConstPtr) {
  RepeatedPtrField<std::RawString>::const_pointer_iterator iter =
      const_proto_array_->pointer_begin();
  static_cast<void>(iter);
}

TEST_F(RepeatedPtrFieldPtrsIteratorTestOnRawString, MutablePtrIteration) {
  RepeatedPtrField<std::RawString>::pointer_iterator iter =
      proto_array_.pointer_begin();
  EXPECT_EQ("foo", (**iter).to_string());
  ++iter;
  EXPECT_EQ("bar", (**(iter)).to_string());
  iter++;
  EXPECT_EQ("baz", (**iter).to_string());
  ++iter;
  EXPECT_TRUE(proto_array_.pointer_end() == iter);
  EXPECT_EQ("baz", (**(--proto_array_.pointer_end())).to_string());
}

TEST_F(RepeatedPtrFieldPtrsIteratorTestOnRawString, MutableConstPtrIteration) {
  RepeatedPtrField<std::RawString>::const_pointer_iterator iter =
      const_proto_array_->pointer_begin();
  EXPECT_EQ("foo", (**iter).to_string());
  ++iter;
  EXPECT_EQ("bar", (**(iter)).to_string());
  iter++;
  EXPECT_EQ("baz", (**iter).to_string());
  ++iter;
  EXPECT_TRUE(const_proto_array_->pointer_end() == iter);
  EXPECT_EQ("baz", (**(--const_proto_array_->pointer_end())).to_string());
}

TEST_F(RepeatedPtrFieldPtrsIteratorTestOnRawString, RandomPtrAccess) {
  RepeatedPtrField<std::RawString>::pointer_iterator iter =
      proto_array_.pointer_begin();
  RepeatedPtrField<std::RawString>::pointer_iterator iter2 = iter;
  ++iter2;
  ++iter2;
  EXPECT_TRUE(iter + 2 == iter2);
  EXPECT_TRUE(iter == iter2 - 2);
  EXPECT_EQ("baz", (*iter[2]).to_string());
  EXPECT_EQ("baz", (**(iter + 2)).to_string());
  EXPECT_EQ(3, proto_array_.end() - proto_array_.begin());
}

TEST_F(RepeatedPtrFieldPtrsIteratorTestOnRawString, RandomConstPtrAccess) {
  RepeatedPtrField<std::RawString>::const_pointer_iterator iter =
      const_proto_array_->pointer_begin();
  RepeatedPtrField<std::RawString>::const_pointer_iterator iter2 = iter;
  ++iter2;
  ++iter2;
  EXPECT_TRUE(iter + 2 == iter2);
  EXPECT_TRUE(iter == iter2 - 2);
  EXPECT_EQ("baz", (*iter[2]).to_string());
  EXPECT_EQ("baz", (**(iter + 2)).to_string());
  EXPECT_EQ(3, const_proto_array_->end() - const_proto_array_->begin());
}

TEST_F(RepeatedPtrFieldPtrsIteratorTestOnRawString, ComparablePtr) {
  RepeatedPtrField<std::RawString>::pointer_iterator iter =
      proto_array_.pointer_begin();
  RepeatedPtrField<std::RawString>::pointer_iterator iter2 = iter + 1;
  EXPECT_TRUE(iter == iter);
  EXPECT_TRUE(iter != iter2);
  EXPECT_TRUE(iter < iter2);
  EXPECT_TRUE(iter <= iter2);
  EXPECT_TRUE(iter <= iter);
  EXPECT_TRUE(iter2 > iter);
  EXPECT_TRUE(iter2 >= iter);
  EXPECT_TRUE(iter >= iter);
}

TEST_F(RepeatedPtrFieldPtrsIteratorTestOnRawString, ComparableConstPtr) {
  RepeatedPtrField<std::RawString>::const_pointer_iterator iter =
      const_proto_array_->pointer_begin();
  RepeatedPtrField<std::RawString>::const_pointer_iterator iter2 = iter + 1;
  EXPECT_TRUE(iter == iter);
  EXPECT_TRUE(iter != iter2);
  EXPECT_TRUE(iter < iter2);
  EXPECT_TRUE(iter <= iter2);
  EXPECT_TRUE(iter <= iter);
  EXPECT_TRUE(iter2 > iter);
  EXPECT_TRUE(iter2 >= iter);
  EXPECT_TRUE(iter >= iter);
}

// Uninitialized iterator does not point to any of the RepeatedPtrOverPtrs.
// Dereferencing an uninitialized iterator crashes the process.
TEST_F(RepeatedPtrFieldPtrsIteratorTestOnRawString, UninitializedPtrIterator) {
  RepeatedPtrField<std::RawString>::pointer_iterator iter;
  EXPECT_TRUE(iter != proto_array_.pointer_begin());
  EXPECT_TRUE(iter != proto_array_.pointer_begin() + 1);
  EXPECT_TRUE(iter != proto_array_.pointer_begin() + 2);
  EXPECT_TRUE(iter != proto_array_.pointer_begin() + 3);
  EXPECT_TRUE(iter != proto_array_.pointer_end());
}

TEST_F(RepeatedPtrFieldPtrsIteratorTestOnRawString, UninitializedConstPtrIterator) {
  RepeatedPtrField<std::RawString>::const_pointer_iterator iter;
  EXPECT_TRUE(iter != const_proto_array_->pointer_begin());
  EXPECT_TRUE(iter != const_proto_array_->pointer_begin() + 1);
  EXPECT_TRUE(iter != const_proto_array_->pointer_begin() + 2);
  EXPECT_TRUE(iter != const_proto_array_->pointer_begin() + 3);
  EXPECT_TRUE(iter != const_proto_array_->pointer_end());
}

TEST_F(RepeatedPtrFieldPtrsIteratorTestOnRawString, PtrMutation) {
  RepeatedPtrField<std::RawString>::pointer_iterator iter =
      proto_array_.pointer_begin();
  **iter = "qux";
  EXPECT_EQ("qux", proto_array_.Get(0).to_string());

  EXPECT_EQ("bar", proto_array_.Get(1).to_string());
  EXPECT_EQ("baz", proto_array_.Get(2).to_string());
  ++iter;
  delete *iter;
  *iter = new std::RawString("a");
  ++iter;
  delete *iter;
  *iter = new std::RawString("b");
  EXPECT_EQ("a", proto_array_.Get(1).to_string());
  EXPECT_EQ("b", proto_array_.Get(2).to_string());
}

// -----------------------------------------------------------------------------
// Unit-tests for the insert iterators
// google::protobuf::RepeatedFieldBackInserter,
// google::protobuf::AllocatedRepeatedPtrFieldBackInserter
// Ported from util/gtl/proto-array-iterators_unittest.

class RepeatedFieldInsertionIteratorsTestOnRawString : public testing::Test {
 protected:
  std::vector<std::RawString> raws;
  TestAllTypes protobuffer;

  virtual void SetUp() {

    raws.push_back(std::RawString("raw"));
    raws.push_back(std::RawString("string"));
    std::copy(raws.begin(), raws.end(),
                RepeatedFieldBackInserter(protobuffer.mutable_repeated_rawstring()));
  }

  virtual void TearDown() {
  }
};

TEST_F(RepeatedFieldInsertionIteratorsTestOnRawString, Raws) {
  ASSERT_EQ(raws.size(), protobuffer.repeated_rawstring_size());
  for (int i = 0; i < raws.size(); ++i)
    EXPECT_EQ(raws.at(i), protobuffer.repeated_rawstring(i));
}

TEST_F(RepeatedFieldInsertionIteratorsTestOnRawString, Raws2) {
  raws.clear();
  raws.push_back("sing");
  raws.push_back("a");
  raws.push_back("song");
  raws.push_back("of");
  raws.push_back("six");
  raws.push_back("pence");
  protobuffer.mutable_repeated_rawstring()->Clear();
  std::copy(
      raws.begin(), raws.end(),
      RepeatedPtrFieldBackInserter(protobuffer.mutable_repeated_rawstring()));
  ASSERT_EQ(raws.size(), protobuffer.repeated_rawstring_size());
  for (int i = 0; i < raws.size(); ++i)
    EXPECT_EQ(raws.at(i), protobuffer.repeated_rawstring(i));
}

TEST_F(RepeatedFieldInsertionIteratorsTestOnRawString,
       AllocatedRepeatedPtrFieldWithString) {
  std::vector<std::RawString*> data;
  TestAllTypes goldenproto;
  for (int i = 0; i < 10; ++i) {
    std::string value = "name-" + StrCat(i);
    std::RawString* new_data = new std::RawString(value);
    data.push_back(new_data);

    auto new_data2 = goldenproto.add_repeated_rawstring();
    new_data2->assign(*new_data);
  }
  TestAllTypes testproto;
  std::copy(data.begin(), data.end(),
            AllocatedRepeatedPtrFieldBackInserter(
                testproto.mutable_repeated_rawstring()));
  EXPECT_EQ(testproto.DebugString(), goldenproto.DebugString());
}

TEST_F(RepeatedFieldInsertionIteratorsTestOnRawString,
       UnsafeArenaAllocatedRepeatedPtrFieldWithString) {
  std::vector<std::RawString*> data;
  TestAllTypes goldenproto;
  for (int i = 0; i < 10; ++i) {
    std::string value = "name-" + StrCat(i);
    std::RawString* new_data = new std::RawString(value);
    data.push_back(new_data);

    auto new_data2 = goldenproto.add_repeated_rawstring();
    new_data2->assign(*new_data);
  }
  TestAllTypes testproto;
  std::copy(data.begin(), data.end(),
            UnsafeArenaAllocatedRepeatedPtrFieldBackInserter(
                testproto.mutable_repeated_rawstring()));
  EXPECT_EQ(testproto.DebugString(), goldenproto.DebugString());
}

TEST_F(RepeatedFieldInsertionIteratorsTestOnRawString, MoveStrings) {
  std::vector<std::RawString> src = {"a", "b", "c", "d"};
  std::vector<std::RawString> copy =
      src;  // copy since move leaves in undefined state
  TestAllTypes testproto;
  std::move(copy.begin(), copy.end(),
            RepeatedFieldBackInserter(testproto.mutable_repeated_rawstring()));

  ASSERT_THAT(testproto.repeated_rawstring(), testing::ElementsAreArray(src));
}

}  // namespace

}  // namespace protobuf
}  // namespace google
