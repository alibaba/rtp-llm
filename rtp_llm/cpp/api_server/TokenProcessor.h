#pragma once

#include <any>

#include "rtp_llm/cpp/utils/PyUtils.h"
#include "rtp_llm/cpp/dataclass/Query.h"
#include "rtp_llm/cpp/api_server/TokenizerEncodeResponse.h"

namespace rtp_llm {

namespace th = torch;

class TokenProcessorPerStream;

class TokenProcessor {
public:
    TokenProcessor(py::object token_processor);
    virtual ~TokenProcessor() = default;

public:
    // virtual for test
    virtual std::string decode(const std::vector<int>& token_ids);
    virtual std::vector<int> encode(const std::string& prompt);
    virtual std::shared_ptr<TokenizerEncodeResponse> tokenizer(const std::string& prompt);
    virtual std::vector<std::string> decodeTokens(std::shared_ptr<TokenProcessorPerStream> tps,
                                                  GenerateOutputs& responses,
                                                  std::vector<int>& output_lens,
                                                  std::shared_ptr<GenerateConfig> config);
    py::object getPyObject() const;

    virtual std::shared_ptr<TokenProcessorPerStream>
    getTokenProcessorCtx(int num_beams,
                         int size,
                         const std::shared_ptr<TokenProcessor>& token_processor_cpp);

private:
    // TODO: change to tokenizer wrapper
    py::object token_processor_;
};

class TokenProcessorPerStream {
public:
    TokenProcessorPerStream() = default;
    ~TokenProcessorPerStream();
    void init(int num_beams, int size, const std::shared_ptr<TokenProcessor>& token_processor_cpp);
    std::vector<std::string> decodeTokens(GenerateOutputs& responses,
                                          std::vector<int>& output_lens,
                                          std::shared_ptr<GenerateConfig> config);
private:
    py::object token_processor_stream_;
};

}  // namespace rtp_llm
