#pragma once

#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/components/QueryManager.h"
#include "maga_transformer/cpp/components/QueryAssembler.h"
#include "maga_transformer/cpp/components/Executor.h"

namespace rtp_llm {

class MagaOp : public th::jit::CustomClassHolder {
public:
    MagaOp(const MagaInitParams& maga_init_params);
    ~MagaOp();

    th::intrusive_ptr<MagaQuery> forward(th::intrusive_ptr<QueryRequest> query);

private:
    void model_loop();

private:
    std::unique_ptr<QueryManager>   query_manager_;
    std::unique_ptr<QueryAssembler> query_assembler_;
    std::unique_ptr<Executor>       executor_;
};

} // namespace rtp_llm

