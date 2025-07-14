#include <gtest/gtest.h>
#include <torch/torch.h>
#include "rtp_llm/cpp/core/QBuffer.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

using namespace std;
using namespace rtp_llm;

class QBufferTest: public ::testing::Test {
public:
    void SetUp() {};
    void TearDown() {};
};

TEST_F(QBufferTest, ValidConstructTest) {
    auto kernel  = BufferPtr(new Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INT8, {10, 10, 10}, (void*)12345));
    auto scales  = BufferPtr(new Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_FP16, {10}, (void*)23456));
    auto zeros   = BufferPtr(new Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_FP16, {10}, (void*)34567));
    auto qbuffer = BufferPtr(new QBuffer(std::move(kernel), std::move(scales), std::move(zeros)));

    // buffer method check
    EXPECT_EQ(qbuffer->isQBuffer(), true);
    EXPECT_EQ(qbuffer->where(), MemoryType::MEMORY_CPU);
    EXPECT_EQ(qbuffer->type(), DataType::TYPE_QINT8);
    EXPECT_EQ((int64_t)qbuffer->data(), 12345);
    EXPECT_EQ((int64_t)qbuffer->dataWithOffset(5), 12350);
    EXPECT_EQ(qbuffer->typeSize(), 1);
    EXPECT_EQ(qbuffer->size(), 10 * 10 * 10);
    EXPECT_EQ(qbuffer->sizeBytes(), 10 * 10 * 10 * 1);
    EXPECT_EQ(qbuffer->dim(), 3);

    // qbuffer method check
    QBufferPtr qbuffer_ptr = std::dynamic_pointer_cast<QBuffer>(qbuffer);
    EXPECT_EQ((int64_t)qbuffer_ptr->data(), 12345);
    EXPECT_EQ((int64_t)qbuffer_ptr->scalesData(), 23456);
    EXPECT_EQ((int64_t)qbuffer_ptr->zerosData(), 34567);
    EXPECT_EQ((int64_t)qbuffer_ptr->scalesType(), DataType::TYPE_FP16);
    EXPECT_EQ((int64_t)qbuffer_ptr->zerosType(), DataType::TYPE_FP16);
    EXPECT_EQ((int64_t)qbuffer_ptr->scalesSizebytes(), 10 * 2);
    EXPECT_EQ((int64_t)qbuffer_ptr->zerosSizebytes(), 10 * 2);
}

TEST_F(QBufferTest, CopyConstructTest) {
    auto kernel = BufferPtr(new Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INT8, {10, 10, 10}, (void*)12345));
    auto scales = BufferPtr(new Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_FP16, {10}, (void*)23456));
    auto zeros  = BufferPtr(new Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_FP16, {10}, (void*)34567));
    EXPECT_EQ((int64_t)kernel->data(), 12345);
    EXPECT_EQ((int64_t)scales->data(), 23456);
    EXPECT_EQ((int64_t)zeros->data(), 34567);
    auto qbuffer = BufferPtr(new QBuffer(std::move(kernel), std::move(scales), std::move(zeros)));
    EXPECT_EQ(kernel, nullptr);
    EXPECT_EQ(scales, nullptr);
    EXPECT_EQ(zeros, nullptr);
}

TEST_F(QBufferTest, Destructor_Test) {
    auto deleter_kernel = [](Buffer* buffer) { std::cout << "delete buffer: kernel" << std::endl; };
    auto kernel =
        BufferPtr(new Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INT8, {10, 10, 10}, nullptr, deleter_kernel));
    auto deleter_scales = [](Buffer* buffer) { std::cout << "delete buffer: scales" << std::endl; };
    auto scales = BufferPtr(new Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INT8, {10}, nullptr, deleter_scales));
    auto deleter_zeros = [](Buffer* buffer) { std::cout << "delete buffer: zeros" << std::endl; };
    auto zeros   = BufferPtr(new Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_INT8, {10}, nullptr, deleter_zeros));
    auto qbuffer = BufferPtr(new QBuffer(std::move(kernel), std::move(scales), std::move(zeros)));
    testing::internal::CaptureStdout();
    qbuffer.reset();
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "delete buffer: zeros\ndelete buffer: scales\ndelete buffer: kernel\n");
}

TEST_F(QBufferTest, TorchConstructTest) {
    auto tensor  = torch::rand({10, 10}).to(torch::kInt8);
    auto scales  = torch::rand({10});
    auto zeros   = torch::randint(0, 10, {10}, at::TensorOptions().dtype(at::ScalarType::Int));
    auto qbuffer = torchTensor2Buffer(tensor, scales, zeros);
    EXPECT_TRUE(dynamic_pointer_cast<QBuffer>(qbuffer)->isQBuffer());
    auto result = QBuffer2torchTensor(dynamic_pointer_cast<const QBuffer>(qbuffer));
    EXPECT_TRUE(torch::equal(result[0], tensor));
    EXPECT_TRUE(torch::equal(result[1], scales));
    EXPECT_TRUE(torch::equal(result[2], zeros));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
