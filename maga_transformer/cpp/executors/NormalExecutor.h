#pragma once

#include "maga_transformer/cpp/deprecated/ParallelModelWrapper.h"
#include "maga_transformer/cpp/deprecated/FakeSampler.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/executors/Executor.h"
#include <memory>
namespace rtp_llm {

class NormalExecutor: public Executor {
public:
    explicit NormalExecutor(const MagaInitParams&                                                   params,
                            const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                            const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights);
    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;

private:
    ModelRequest generateOldModelRequest(GptModelInputs& model_input);

private:
    std::unique_ptr<ParallelModelWrapper> model_wrapper_;
    std::unique_ptr<FakeSampler>          fake_sampler_;
};

}  // namespace rtp_llm
