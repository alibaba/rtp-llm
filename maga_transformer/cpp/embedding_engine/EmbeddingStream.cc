#include "maga_transformer/cpp/embedding_engine/EmbeddingStream.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include <memory>

using namespace std;

namespace rtp_llm {

EmbeddingStream::EmbeddingStream(const shared_ptr<rtp_llm::EmbeddingInput>& query):
    embedding_input_(query){
    if (!query.get()) {
        return;
    }
    begin_time_         = TimeUtility::currentTimeInMilliSeconds();
    device_             = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    embedding_output_ = make_shared<EmbeddingOutput>();
    generate_state_ = GenerateState::WAITING;
}

int64_t EmbeddingStream::streamId() const {
    return embedding_input_->request_id;
}

int64_t EmbeddingStream::batchSize() const {
    return embedding_input_->input_lengths->shape()[0];
}

std::shared_ptr<EmbeddingInput> EmbeddingStream::embeddingInput() const{
    return embedding_input_;
}

std::shared_ptr<EmbeddingOutput> EmbeddingStream::embeddingOutput() const{
    return embedding_output_;
}

int64_t EmbeddingStream::inputLength() const {
    return embedding_input_->total_length;
}

void EmbeddingStream::waitFinish() {
    unique_lock<mutex> lock(lock_);
    while (generate_state_ != GenerateState::FINISHED && generate_state_ != GenerateState::STOPPED) {    
        cond_.wait_for(lock, std::chrono::milliseconds(5));
    }
    if (embedding_output_->error_info.has_error) {
        throw std::runtime_error("run stream failed: " + embedding_output_->error_info.error_message);
    }
}

void EmbeddingStream::setError(const std::string& error_info) {
    embedding_output_->setError(error_info);
    generate_state_ = GenerateState::STOPPED;
}

void EmbeddingStream::updateOutput(ft::BufferPtr& output) {
    lock_guard<mutex> lock(lock_);
    embedding_output_->setOutput(output, 0);
    generate_state_ = GenerateState::FINISHED;
    cond_.notify_all();
}

} // namespace rtp_llm