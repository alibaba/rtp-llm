#include <gtest/gtest.h>
#include <torch/torch.h>
#include "src/fastertransformer/core/QBuffer.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

using namespace std;
using namespace fastertransformer;

class QBufferTest: public ::testing::Test {
public:
    void SetUp() {};
    void TearDown() {};
};

TEST_F(QBufferTest, ValidConstructTest) {
    auto kernel = Buffer(MemoryType::MEMORY_CPU,
                         DataType::TYPE_INT8,
                         {10, 10, 10},
                         (void*)12345);
    auto scales = Buffer(MemoryType::MEMORY_CPU,
                         DataType::TYPE_FP16,
                         {10},
                         (void*)23456);
    auto zeros = Buffer(MemoryType::MEMORY_CPU,
                        DataType::TYPE_FP16,
                        {10},
                        (void*)34567);
    Buffer* qbuffer = new QBuffer(std::move(kernel),
                                  std::move(scales),
                                  std::move(zeros));
    // buffer method check
    EXPECT_EQ(qbuffer->isQuantify(), true);
    EXPECT_EQ(qbuffer->where(), MemoryType::MEMORY_CPU);
    EXPECT_EQ(qbuffer->type(), DataType::TYPE_QINT8);
    EXPECT_EQ((int64_t)qbuffer->data(), 12345);
    EXPECT_EQ((int64_t)qbuffer->dataWithOffset(5), 12350);
    EXPECT_EQ(qbuffer->deleter(), nullptr);
    EXPECT_EQ(qbuffer->typeSize(), 1);
    EXPECT_EQ(qbuffer->size(), 10 * 10 * 10);
    EXPECT_EQ(qbuffer->sizeBytes(), 10 * 10 * 10 * 1);
    EXPECT_EQ(qbuffer->dim(), 3);

    // qbuffer method check
    QBuffer* qbuffer_ptr= dynamic_cast<QBuffer*>(qbuffer);
    EXPECT_EQ((int64_t)qbuffer_ptr->data(), 12345);
    EXPECT_EQ((int64_t)qbuffer_ptr->scales_data(), 23456);
    EXPECT_EQ((int64_t)qbuffer_ptr->zeros_data(), 34567);
    EXPECT_EQ((int64_t)qbuffer_ptr->scales_type(), DataType::TYPE_FP16);
    EXPECT_EQ((int64_t)qbuffer_ptr->zeros_type(), DataType::TYPE_FP16);
    EXPECT_EQ((int64_t)qbuffer_ptr->scales_sizeBytes(), 10 * 2);
    EXPECT_EQ((int64_t)qbuffer_ptr->zeros_sizeBytes(), 10 * 2);
}

TEST_F(QBufferTest, CopyConstructTest) {
    auto kernel = Buffer(MemoryType::MEMORY_CPU,
                         DataType::TYPE_INT8,
                         {10, 10, 10},
                         (void*)12345);
    auto scales = Buffer(MemoryType::MEMORY_CPU,
                         DataType::TYPE_FP16,
                         {10},
                         (void*)23456);
    auto zeros = Buffer(MemoryType::MEMORY_CPU,
                        DataType::TYPE_FP16,
                        {10},
                        (void*)34567);
    EXPECT_EQ((int64_t)kernel.data(), 12345);
    EXPECT_EQ((int64_t)scales.data(), 23456);
    EXPECT_EQ((int64_t)zeros.data(), 34567);
    Buffer qbuffer = QBuffer(std::move(kernel),
                             std::move(scales),
                             std::move(zeros));
    EXPECT_EQ((int64_t)kernel.data(), 0);
    EXPECT_EQ((int64_t)scales.data(), 0);
    EXPECT_EQ((int64_t)zeros.data(), 0);
}

TEST_F(QBufferTest, Destructor_Test) {
    auto deleter_kernel = [] (Buffer* buffer) {
        std::cout << "delete buffer: kernel" << std::endl;};
    auto kernel = new Buffer(MemoryType::MEMORY_CPU,
                            DataType::TYPE_INT8,
                            {10, 10, 10},
                            nullptr,
                            deleter_kernel);
    auto deleter_scales = [] (Buffer* buffer) {
        std::cout << "delete buffer: scales" << std::endl;};
    auto scales = new Buffer(MemoryType::MEMORY_CPU,
                            DataType::TYPE_INT8,
                            {10},
                            nullptr,
                            deleter_scales);
    auto deleter_zeros = [] (Buffer* buffer) {
        std::cout << "delete buffer: zeros" << std::endl;};
    auto zeros = new Buffer(MemoryType::MEMORY_CPU,
                            DataType::TYPE_INT8,
                            {10},
                            nullptr,
                            deleter_zeros);
    Buffer* qbuffer = new QBuffer(std::move(*kernel),
                                  std::move(*scales),
                                  std::move(*zeros));
    testing::internal::CaptureStdout();
    delete qbuffer;
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "delete buffer: zeros\ndelete buffer: scales\ndelete buffer: kernel\n");
    
}

TEST_F(QBufferTest, TorchConstructTest) {
    auto tensor = torch::rand({10, 10}).to(torch::kInt8);
    auto scales = torch::rand({10});
    auto zeros  = torch::randint(0, 10, {10}, at::TensorOptions().dtype(at::ScalarType::Int));
    auto qbuffer = torchTensor2Buffer(tensor, scales, zeros);
    EXPECT_TRUE(static_pointer_cast<QBuffer>(qbuffer)->isQuantify());
    auto result = QBuffer2torchTensor(static_pointer_cast<const QBuffer>(qbuffer));
    EXPECT_TRUE(torch::equal(result[0], tensor));
    EXPECT_TRUE(torch::equal(result[1], scales));
    EXPECT_TRUE(torch::equal(result[2], zeros));
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
