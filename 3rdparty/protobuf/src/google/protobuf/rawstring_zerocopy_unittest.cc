#include <google/protobuf/rawstring.h>
#include <google/protobuf/rawstring_zerocopy_unittest.pb.h>
#include <google/protobuf/io/zero_copy_stream_ext.h>
#include <google/protobuf/testing/googletest.h>
#include <gtest/gtest.h>
#include <string.h>
#include <vector>


namespace google {
namespace protobuf {

struct MyMessage {
  struct Iov {
    std::shared_ptr<const char> data;
    uint32_t size;
  };

  std::vector<Iov> iov_list;

  void push_back(const std::shared_ptr<const char>& data, uint32_t size) {
    iov_list.push_back(Iov({data, size}));
  }
};

class MyInputStream : public io::ZeroCopyInputStream4RawStr {
 public:
  MyInputStream(const std::shared_ptr<MyMessage>& message)
      : message_(message) {}

  bool Next(const void** data, int* size) override {
    if (iov_index_for_next_ >= message_->iov_list.size()) {
      printf("MyInputStream::Next() called. return false.\n");
      return false;
    }

    *data = message_->iov_list[iov_index_for_next_].data.get();
    *size = message_->iov_list[iov_index_for_next_].size;

    iov_index_for_next_++;

    printf("MyInputStream::Next() called. return true. data: %p, size: %d\n",
           *data, *size);
    return true;
  }

  void BackUp(int count) override {
    printf("MyInputStream::BackUp() called. count = %d\n", count);
  }

  bool Skip(int count) override {
    printf("MyInputStream::Skip() called, count = %d\n", count);
    return false;
  }

  int64_t ByteCount() const override {
    printf("MyInputStream::ByteCount() called\n");
    return 0;
  }

  bool FillRawString(std::RawString* str, size_t str_size, const char* data,
                     size_t data_size) {
    assert(str->size() == 0);
    assert(str_size == data_size);

    auto last_iov = message_->iov_list[iov_index_for_next_ - 1];
    assert(last_iov.data.get() == data);
    assert(last_iov.size == data_size);

    str->assign(last_iov.data, last_iov.size);
    return true;
  }

 private:
  std::shared_ptr<MyMessage> message_;
  uint32_t iov_index_for_next_{0};
};

TEST(ZeroCopyStreamTest, InputStream) {
  auto message = std::make_shared<MyMessage>();
  {
    // Fill message by serialized message.
    demo::Demo msg;
    msg.set_int32_field(42);
    msg.set_string_field("aloha");
    msg.add_repeated_string_field("good");
    msg.add_repeated_string_field("good");
    msg.add_repeated_string_field("study");

    auto raw_string_data = "How to become batman?";
    auto raw_string_size = strlen(raw_string_data);
    msg.set_raw_string_field(raw_string_data, raw_string_size);

    auto serialized_str = msg.SerializeAsString();

    // Split serialized str to two iov.
    size_t data_1_size = serialized_str.size() - raw_string_size;
    std::shared_ptr<char> data_1(new char[data_1_size]);
    memcpy(data_1.get(), serialized_str.data(), data_1_size);
    message->push_back(data_1, data_1_size);

    std::shared_ptr<char> data_2(new char[raw_string_size]);
    memcpy(data_2.get(), serialized_str.data() + data_1_size, raw_string_size);
    message->push_back(data_2, raw_string_size);
  }

  demo::Demo parsed_msg;
  MyInputStream input(message);
  parsed_msg.ParseFromZeroCopyStream(&input);

  EXPECT_EQ(42, parsed_msg.int32_field());
  EXPECT_EQ("aloha", parsed_msg.string_field());
  EXPECT_EQ(3, parsed_msg.repeated_string_field().size());
  EXPECT_EQ("good", parsed_msg.repeated_string_field(0));
  EXPECT_EQ("good", parsed_msg.repeated_string_field(1));
  EXPECT_EQ("study", parsed_msg.repeated_string_field(2));

  //utils::hexdump(parsed_msg.raw_string_field().data(),
  //               parsed_msg.raw_string_field().size());
  EXPECT_EQ(strlen("How to become batman?"),
            parsed_msg.raw_string_field().size());
  EXPECT_EQ("How to become batman?", parsed_msg.raw_string_field().to_string());
  EXPECT_EQ(message->iov_list[1].data, parsed_msg.raw_string_field().shared_data());
}

class MyOutputStream : public io::ZeroCopyOutputStream4RawStr {
public:
    MyOutputStream(std::shared_ptr<MyMessage>& message)
        : message_(message) {}

public:
    virtual bool Next(void **data, int *size) override {
        if (byteCount_ >= message_->iov_list[0].size) {
            return false;
        }

        *data = const_cast<char*>(message_->iov_list[0].data.get()) + byteCount_;
        *size = message_->iov_list[0].size - byteCount_;

        byteCount_ += *size;
        return true;
    }

    virtual void BackUp(int count) override {
        byteCount_ -= count;
    }

    virtual int64_t ByteCount() const override { return byteCount_; }
    virtual bool SupportRawString() const override { return true; }
    virtual bool WriteRawString(std::RawString *rstr) override {
        message_->push_back(rstr->shared_data(), rstr->size());
        return true;
    }

private:
    int64_t byteCount_;
    std::shared_ptr<MyMessage> message_;
};

TEST(ZeroCopyStreamTest, OutputStream) {
    auto message = std::make_shared<MyMessage>();

    std::shared_ptr<char> data(new char[1024]);
    message->push_back(data, 1024);

    std::RawString rstr1("I am batman");
    std::RawString rstr2("I have super power");
    std::RawString rstr3("I am rich");

    demo::Demo msg;
    msg.set_int32_field(42);
    msg.set_string_field("aloha");
    msg.add_repeated_string_field("good");
    msg.add_repeated_string_field("good");
    msg.add_repeated_string_field("study");
    msg.set_raw_string_field(rstr1);
    auto rstr = msg.add_repeated_raw_string_field();
    rstr->assign(rstr2);
    rstr = msg.add_repeated_raw_string_field();
    rstr->assign(rstr3);

    MyOutputStream stream(message);
    ::google::protobuf::io::CodedOutputStream encoder(&stream);
    encoder.SetSerializationDeterministic(true);

    msg.SerializeToCodedStream(&encoder);

    ASSERT_EQ(4, message->iov_list.size());
    EXPECT_EQ(rstr1.shared_data(), message->iov_list[1].data);
    EXPECT_EQ(rstr1.size(), message->iov_list[1].size);
    EXPECT_EQ(rstr2.shared_data(), message->iov_list[2].data);
    EXPECT_EQ(rstr2.size(), message->iov_list[2].size);
    EXPECT_EQ(rstr3.shared_data(), message->iov_list[3].data);
    EXPECT_EQ(rstr3.size(), message->iov_list[3].size);
}

}
}