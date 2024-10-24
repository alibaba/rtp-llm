#include <google/protobuf/rawstring.h>

namespace std {

RawString::RawString(RawString&& rstr) {
  assign(rstr);
  rstr.clear();
}

RawString::RawString(const string& str) { assign(str); }

RawString::RawString(string&& str) { assign(std::move(str)); }

RawString::RawString(const char* c) { assign(c); }

RawString::RawString(const char* c, size_t size) { assign(c, size); }

RawString::RawString(const char* c, size_t offset, size_t size) {
  assign(c, offset, size);
}

RawString::RawString(const shared_ptr<const char>& ptr, size_t size) { assign(ptr, size); }

RawString::RawString(const shared_ptr<const char>& ptr, size_t offset, size_t size) {
  assign(ptr, offset, size);
}

void RawString::assign(const string& str) {
  clear();
  string *new_str = new string(str);
  set_string_data(new_str);
}

void RawString::assign(string&& str) {
  clear();
  string *new_str = new string(std::move(str));
  set_string_data(new_str);
}

void RawString::assign(const char* c) {
  clear();
  string *new_str = new string(c);
  set_string_data(new_str);
}

void RawString::assign(const char* c, size_t size) {
  assign(c, 0, size);
}

void RawString::assign(const char* c, size_t offset, size_t size) {
  clear();
  string* new_str = new string(c + offset, size);
  set_string_data(new_str);
}

void RawString::assign(const shared_ptr<const char>& ptr, size_t size) {
  assign(ptr, 0, size);
}

void RawString::assign(const shared_ptr<const char>& ptr, size_t offset, size_t size) {
  clear();
  own_data_ = ptr;
  offset_ = offset;
  size_ = size;
}

void RawString::assign(const std::RawString& rstr) {
  clear();
  own_data_ = rstr.own_data_;
  size_ = rstr.size_;
  offset_ = rstr.offset_;
  device_ = rstr.device_;
  remote_device_ = rstr.remote_device_;
}

void RawString::set_string_data(const string* str) {
  shared_ptr<const char> ptr(str->c_str(), [str](const char *p) { delete str; });
  own_data_ = ptr;
  offset_ = 0;
  size_ = str->size();
}

void RawString::swap(RawString& rstr) {
#define do_swap(a, b) \
  do {                \
    auto t = (a);     \
    (a) = (b);        \
    (b) = t;          \
  } while (0) 
  do_swap(own_data_, rstr.own_data_);
  do_swap(size_, rstr.size_);
  do_swap(offset_, rstr.offset_);
  do_swap(device_, rstr.device_);
  do_swap(remote_device_, rstr.remote_device_);
#undef do_swap
}

void RawString::clear() {
  own_data_.reset();
  size_ = 0;
  offset_ = 0;
  device_ = UNKNOWN;
  remote_device_ = UNKNOWN;
}

bool RawString::operator==(const RawString& rstr) const {
  if (this->device() != rstr.device()) {
    return false;
  }
  if (this->remote_device() != rstr.remote_device()) {
    return false;
  }
  if (this->size() != rstr.size()) {
    return false;
  }
  if (this->data() == nullptr && rstr.data() == nullptr) {
    return true;
  }
  if (this->data() == nullptr || rstr.data() == nullptr) {
    return false;
  }
  return memcmp(data(), rstr.data(), this->size()) == 0;
}

string RawString::substr(size_t start, size_t length) const {
  string str = to_string();
  if (start + length > str.length()) {
    return "";
  }

  return str.substr(start, length);
}

string RawString::to_string() const {
  if (own_data_ == nullptr || size_ == 0) {
    return "";
  }
  return string(data(), size_);
}

}  // namespace std
