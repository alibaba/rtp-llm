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

#include <iostream>
#include <vector>

#include "abstract_decoder.h"
#include "abstract_searcher.h"
#include "dtype.h"

namespace xft {
class Model {
public:
    Model();
    ~Model();

    void input(std::vector<int32_t> &inputIds_, int batchSize_);

    void config(int maxLen_ = -1, int numBeams_ = 1, int numBeamHypsToKeep_ = 1, float lenPenalty_ = 1.0,
            bool doEarlyStopping_ = false, int eosTokenId_ = -1, int padTokenId_ = -1, bool doSample_ = false,
            float temperature_ = 1.0, int topK_ = 50, float topP_ = 1.0, float repetitionPenalty_ = 1.0,
            const std::vector<std::vector<int>> &stopWordsList_ = {});

    void config(SearcherConfig &config_, const std::vector<std::vector<int>> &stopWordsList_ = {});

    bool isDone();

    std::tuple<float *, int, int> forward();

    std::vector<int32_t> generate();

    void createSearcher(SearcherConfig &config_);

    bool isMaster();

    int getRank();

    int getBatchSize() { return batchSize; }

    int getSeqLen() { return seqLen; }

    void setVocabSize(int vocabSize) { this->vocabSize = vocabSize; }

    int getVocabSize() { return this->vocabSize; }

    SearcherConfig getConfig() { return configuration; }

    void setDecoder(AbstractDecoder *dec);

    std::vector<int32_t> finalize() { return searcher->finalize(); }

    void exitSlaves();

    void setPrefix(std::vector<int32_t> &prefixIDs);

    void unsetPrefix();

    bool setStopWords(std::vector<std::vector<int>> stopWordsList);

private:
    AbstractDecoder *decoder;
    AbstractSearcher *searcher;
    std::vector<int32_t> inputIds;
    int batchSize;
    int seqLen;
    int vocabSize;
    SearcherConfig configuration;
    bool isNewInput;
};

class AutoModel : public Model {
public:
    AutoModel(std::string modelPath, xft::DataType dataType, xft::DataType KVCacheDataType = xft::DataType::fp16);
};
} // namespace xft