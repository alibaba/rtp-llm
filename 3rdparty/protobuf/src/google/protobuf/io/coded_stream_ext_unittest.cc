#include "google/protobuf/io/coded_stream_ext.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_ext.h"

using testing::_;
using testing::InSequence;
using testing::NiceMock;
using testing::Return;
using testing::StrictMock;

namespace google {
namespace protobuf {
namespace io {

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

class MockZeroCopyInputStream : public ZeroCopyInputStream {
 public:
  virtual ~MockZeroCopyInputStream() = default;
  MOCK_METHOD2(Next, bool(const void** data, int* size));
  MOCK_METHOD1(BackUp, void(int count));
  MOCK_METHOD1(Skip, bool(int count));
  MOCK_CONST_METHOD0(ByteCount, int64());
};

class MockZeroCopyInputStream4RawStr : public ZeroCopyInputStream4RawStr {
 public:
  virtual ~MockZeroCopyInputStream4RawStr() = default;
  MOCK_METHOD2(Next, bool(const void** data, int* size));
  MOCK_METHOD1(BackUp, void(int count));
  MOCK_METHOD1(Skip, bool(int count));
  MOCK_CONST_METHOD0(ByteCount, int64());
  MOCK_METHOD4(FillRawString, bool(std::RawString* str, size_t str_size,
                                   const char* data, size_t data_size));
};

class MockZeroCopyOutputStream : public ZeroCopyOutputStream {
 public:
  virtual ~MockZeroCopyOutputStream() = default;

  MOCK_METHOD2(Next, bool(void** data, int* size));
  MOCK_METHOD1(BackUp, void(int count));
  MOCK_CONST_METHOD0(ByteCount, int64());
  MOCK_METHOD2(WriteAliasedRaw, bool(const void* data, int size));
  MOCK_CONST_METHOD0(AllowsAliasing, bool());
};

class MockZeroCopyOutputStream4RawStr : public ZeroCopyOutputStream4RawStr {
 public:
  virtual ~MockZeroCopyOutputStream4RawStr() = default;

  MOCK_METHOD2(Next, bool(void** data, int* size));
  MOCK_METHOD1(BackUp, void(int count));
  MOCK_CONST_METHOD0(ByteCount, int64());
  MOCK_METHOD2(WriteAliasedRaw, bool(const void* data, int size));
  MOCK_CONST_METHOD0(AllowsAliasing, bool());

  MOCK_CONST_METHOD0(SupportRawString, bool());
  MOCK_METHOD1(WriteRawString, bool(std::RawString* rstr));
};

TEST(CodedStreamExtHelperTest, ReadRawStringWithNormalInput_OneNextCall) {
  NiceMock<MockZeroCopyInputStream> input;
  auto coded_in = make_unique<CodedInputStream>(&input);

  EXPECT_CALL(input, Next(_, _))
      .WillOnce([](const void** data, int* size) -> bool {
        *data = "hello";
        *size = 5;
        return true;
      });

  std::RawString rawstring;
  bool success =
      CodedStreamExtHelper::ReadRawString(coded_in.get(), &rawstring, 5);
  EXPECT_TRUE(success);

  // When a RawString is read from normal input, a std::string will be created
  // and set to the RawString, and the RawString should have ownership of the
  // std::string.
  EXPECT_STREQ("hello", rawstring.data());
}

TEST(CodedStreamExtHelperTest, ReadRawStringWithNormalInput_MultiNextCall) {
  NiceMock<MockZeroCopyInputStream> input;
  auto coded_in = make_unique<CodedInputStream>(&input);

  {
    // If the Next() method does not return enough data, the Next() method
    // will be called again.
    InSequence s;

    EXPECT_CALL(input, Next(_, _))
        .WillOnce([](const void** data, int* size) -> bool {
          *data = "hello";
          *size = 5;
          return true;
        });

    EXPECT_CALL(input, Next(_, _))
        .WillOnce([](const void** data, int* size) -> bool {
          *data = " world";
          *size = 6;
          return true;
        });
  }

  std::RawString rawstring;
  bool success =
      CodedStreamExtHelper::ReadRawString(coded_in.get(), &rawstring, 11);
  EXPECT_TRUE(success);

  EXPECT_STREQ("hello world", rawstring.data());
}

TEST(CodedStreamExtHelperTest, ReadRawStringWithNormalInput_BackUp) {
  NiceMock<MockZeroCopyInputStream> input;
  auto coded_in = make_unique<CodedInputStream>(&input);

  {
    // If the Next() method return too much data
    InSequence s;

    EXPECT_CALL(input, Next(_, _))
        .WillOnce([](const void** data, int* size) -> bool {
          *data = "hello";
          *size = 5;
          return true;
        });

    EXPECT_CALL(input, Next(_, _))
        .WillOnce([](const void** data, int* size) -> bool {
          *data = " world xxx";
          *size = 10;
          return true;
        });
  }

  std::RawString rawstring;
  bool success =
      CodedStreamExtHelper::ReadRawString(coded_in.get(), &rawstring, 11);
  EXPECT_TRUE(success);

  EXPECT_STREQ("hello world", rawstring.data());

  // When CodedInputStream is destructed, the BackUp() method should be called
  // to reclaim the excess memory.
  EXPECT_CALL(input, BackUp).WillOnce([](int count) { EXPECT_EQ(4, count); });
  coded_in.reset(nullptr);
}

TEST(CodedStreamExtHelperTest, ReadRawStringWithRawStrInput_OneNextCall) {
  NiceMock<MockZeroCopyInputStream4RawStr> input;
  auto coded_in = make_unique<CodedInputStream>(&input);

  std::string input_data = "hello";

  EXPECT_CALL(input, Next(_, _))
      .WillOnce([&](const void** data, int* size) -> bool {
        *data = input_data.c_str();
        *size = input_data.size();
        return true;
      });

  std::RawString rawstring;
  EXPECT_CALL(input, FillRawString)
      .WillOnce([&](std::RawString* str, size_t str_size, const char* data,
                    size_t data_size) -> bool {
        EXPECT_EQ(&rawstring, str);
        EXPECT_EQ(input_data.size(), str_size);
        EXPECT_EQ(input_data.data(), data);
        EXPECT_EQ(input_data.size(), data_size);
        auto shared = std::shared_ptr<const char>(data, [data](const char*) {});
        str->assign(shared, data_size);
        return true;
      });

  bool success = CodedStreamExtHelper::ReadRawString(
      coded_in.get(), &rawstring, input_data.size());
  EXPECT_TRUE(success);

  EXPECT_EQ(input_data.c_str(), rawstring.data());
}

TEST(CodedStreamExtHelperTest, ReadRawStringWithRawStrInput_MultiNextCall) {
  NiceMock<MockZeroCopyInputStream4RawStr> input;
  auto coded_in = make_unique<CodedInputStream>(&input);

  std::string input_data_1 = "hello";
  std::string input_data_2 = " world";

  {
    InSequence s;

    EXPECT_CALL(input, Next(_, _))
        .WillOnce([&](const void** data, int* size) -> bool {
          *data = input_data_1.c_str();
          *size = input_data_1.size();
          return true;
        });

    EXPECT_CALL(input, Next(_, _))
        .WillOnce([&](const void** data, int* size) -> bool {
          *data = input_data_2.c_str();
          *size = input_data_2.size();
          return true;
        });
  }

  std::RawString rawstring;
  {
    InSequence s;

    EXPECT_CALL(input, FillRawString)
        .WillOnce([&](std::RawString* str, size_t str_size, const char* data,
                      size_t data_size) -> bool {
          EXPECT_EQ(&rawstring, str);
          EXPECT_EQ(11, str_size);
          EXPECT_EQ(input_data_1.data(), data);
          EXPECT_EQ(input_data_1.size(), data_size);

          str->assign(data, data_size);
          return true;
        });
    EXPECT_CALL(input, FillRawString)
        .WillOnce([&](std::RawString* str, size_t str_size, const char* data,
                      size_t data_size) -> bool {
          EXPECT_EQ(&rawstring, str);
          EXPECT_EQ(11, str_size);
          EXPECT_EQ(input_data_2.data(), data);
          EXPECT_EQ(input_data_2.size(), data_size);

          str->assign(str->to_string() + std::string(data, data_size));
          return true;
        });
  }

  bool success =
      CodedStreamExtHelper::ReadRawString(coded_in.get(), &rawstring, 11);
  EXPECT_TRUE(success);

  EXPECT_EQ(input_data_1.size() + input_data_2.size(), rawstring.size());
  EXPECT_EQ("hello world", rawstring.to_string());
}

TEST(CodedStreamExtHelperTest, WriteRawStringWithNormalOutput_OneNextCall) {
  NiceMock<MockZeroCopyOutputStream> output;
  CodedOutputStream coded_out(&output);

  char output_buf[12] = {0};

  EXPECT_CALL(output, Next(_, _)).WillOnce([&](void** data, int* size) -> bool {
    *data = output_buf;
    *size = 11;
    return true;
  });

  std::RawString rawstring("hello world");
  bool success = CodedStreamExtHelper::WriteRawString(&coded_out, &rawstring);
  EXPECT_TRUE(success);

  EXPECT_STREQ("hello world", output_buf);
}

TEST(CodedStreamExtHelperTest, WriteRawStringWithNormalOutput_MultiNextCall) {
  NiceMock<MockZeroCopyOutputStream> output;
  CodedOutputStream coded_out(&output);

  char output_buf[12] = {0};

  {
    InSequence s;

    EXPECT_CALL(output, Next(_, _))
        .WillOnce([&](void** data, int* size) -> bool {
          *data = output_buf;
          *size = 5;
          return true;
        });

    EXPECT_CALL(output, Next(_, _))
        .WillOnce([&](void** data, int* size) -> bool {
          *data = output_buf + 5;
          *size = 6;
          return true;
        });
  }

  std::RawString rawstring("hello world");
  bool success = CodedStreamExtHelper::WriteRawString(&coded_out, &rawstring);
  EXPECT_TRUE(success);

  EXPECT_STREQ("hello world", output_buf);
}

TEST(CodedStreamExtHelperTest, WriteRawStringWithNormalOutput_BackUp) {
  NiceMock<MockZeroCopyOutputStream> output;
  auto coded_out = make_unique<CodedOutputStream>(&output);

  char output_buf[16] = {0};

  {
    InSequence s;

    EXPECT_CALL(output, Next(_, _))
        .WillOnce([&](void** data, int* size) -> bool {
          *data = output_buf;
          *size = 5;
          return true;
        });

    EXPECT_CALL(output, Next(_, _))
        .WillOnce([&](void** data, int* size) -> bool {
          *data = output_buf + 5;
          *size = 10;
          return true;
        });
  }

  std::RawString rawstring("hello world");
  bool success =
      CodedStreamExtHelper::WriteRawString(coded_out.get(), &rawstring);
  EXPECT_TRUE(success);
  EXPECT_STREQ("hello world", output_buf);

  // When CodedOutputStream is destructed, the BackUp() method should be called
  // to reclaim the excess memory.
  EXPECT_CALL(output, BackUp).WillOnce([](int count) { EXPECT_EQ(4, count); });
  coded_out.reset(nullptr);
}

TEST(CodedStreamExtHelperTest, WriteRawStringWithRawStrOutput) {
  NiceMock<MockZeroCopyOutputStream4RawStr> output;
  CodedOutputStream coded_out(&output);

  ON_CALL(output, SupportRawString).WillByDefault(Return(true));

  EXPECT_CALL(output, Next(_, _)).Times(0);
  EXPECT_CALL(output, BackUp(_)).WillOnce([](int count) -> bool {
    EXPECT_EQ(0, count);
    return true;
  });
  std::RawString rawstring("hello world");

  EXPECT_CALL(output, WriteRawString(_))
      .WillOnce([&](std::RawString* writed_rawstring) -> bool {
        EXPECT_EQ(&rawstring, writed_rawstring);
        return true;
      });

  bool success = CodedStreamExtHelper::WriteRawString(&coded_out, &rawstring);
  EXPECT_TRUE(success);
}

}  // namespace io
}  // namespace protobuf
}  // namespace google