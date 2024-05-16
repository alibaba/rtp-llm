// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once
#include <cstdint>
#include <tuple>
#include <vector>

class AbstractSearcher {
public:
    virtual ~AbstractSearcher() {}

    // First call to get NextToken, return {batchSize, numBeams}. For
    // greadySearch, numBeams = 1.
    virtual std::vector<int> getNextToken(int *ids, int batchSize, int seqLen) = 0;

    // Subsequent calls to get next Token
    virtual std::vector<int> getNextToken() = 0;

    virtual bool isDone() = 0;

    virtual std::vector<int32_t> finalize() = 0;

    virtual bool setStopWords(std::vector<std::vector<int>> stopWordsList) = 0;
};

struct SearcherConfig {
    bool doEarlyStopping = false;
    bool doSample = false;
    int maxLen = -1;
    int numBeams = 1;
    int numBeamHypsToKeep = 1;
    int eosTokenId = -1;
    int padTokenId = -1;
    int topK = 50;
    float lenPenalty = 1.0;
    float temperature = 1.0;
    float topP = 1.0;
    float repetitionPenalty = 1.0;

    SearcherConfig(int maxLen_ = -1, int numBeams_ = 1, int numBeamHypsToKeep_ = 1, float lenPenalty_ = 1.0,
            bool doEarlyStopping_ = false, int eosTokenId_ = -1, int padTokenId_ = -1, bool doSample_ = false,
            float temperature_ = 1.0, int topK_ = 50, float topP_ = 1.0, float repetitionPenalty_ = 1.0)
        : maxLen(maxLen_)
        , numBeams(numBeams_)
        , numBeamHypsToKeep(numBeamHypsToKeep_)
        , lenPenalty(lenPenalty_)
        , doEarlyStopping(doEarlyStopping_)
        , eosTokenId(eosTokenId_)
        , padTokenId(padTokenId_)
        , doSample(doSample_)
        , temperature(temperature_)
        , topK(topK_)
        , topP(topP_)
        , repetitionPenalty(repetitionPenalty_) {}
};
