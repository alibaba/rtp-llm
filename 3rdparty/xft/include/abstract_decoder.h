// Copyright (c) 2023-2024 Intel Corporation
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

class DecoderContext;
class Messenger;

class AbstractDecoder {
public:
    virtual ~AbstractDecoder() {}

    // Forward function with the input IDs with shape of dims - (batchSize, beamSize, seqLen)
    // Return the decoding result, split offset, and split size
    // The returned result is a split representing the possibilities of next token, like the shadow part in below graph
    //                                         splitOffset
    //                                               \|<-splitSize->|
    //    _                ___________________________v______________________________________
    //    ^               |             |             |||||||||||||||             |          |
    //    |               |             |             |||||||||||||||             |          |
    // batchSize*beamSize |             |             |||||||||||||||             |          |
    //    |               |             |             |||||||||||||||             |          |
    //    v               |_____________|_____________|||||||||||||||_____________|__________|
    //                    |<----------------------- vocabSize  ----------------------------->|
    virtual std::tuple<float *, int, int> forward(int *ids, int64_t *dims, int step, bool logits_all = false) = 0;

    // Reorder cached keys and values, size=batchSize*beamSize
    virtual void reorderCache(int *idx, int size) = 0;

    virtual DecoderContext *getContext() = 0;

    virtual Messenger &getMessenger() = 0;

    virtual bool isMaster() = 0;

    virtual int getRank() = 0;

    virtual int getEndId() = 0;

    virtual void setPrefix(int *ids, int seqLen) = 0;

    virtual void unsetPrefix() = 0;
};
