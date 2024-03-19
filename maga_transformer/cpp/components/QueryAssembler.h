#pragma once

#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/components/QueryManager.h"

namespace rtp_llm {

class QueryAssembler {
public:
    QueryAssembler() {};
    ~QueryAssembler() {};

    std::unique_ptr<MergedRequest> assemble_requests(
        const std::shared_ptr<const QueryGroup> &query_groups) const;

    // Here the outputs are only splitted by query groups for samplers.
    // std::vector<SamplerRequest> disassemble_output(
    //     const std::vector<std::shared_ptr<const QueryGroup>> &query_groups,
    //     const ModelOutput &model_output) const;

private:

};

}
