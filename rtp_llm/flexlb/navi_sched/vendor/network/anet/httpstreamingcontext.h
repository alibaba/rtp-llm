#pragma once
#include <cstring>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/streamingcontext.h"

namespace anet {
class HTTPStreamingContext : public StreamingContext {
public:
    HTTPStreamingContext();
    virtual ~HTTPStreamingContext();

    enum HTTPStreamingStep {
        HSS_START_LINE = 0,
        HSS_MESSAGE_HEADER,
        HSS_MESSAGE_BODY
    };
    enum HTTPEncodingType {
        HET_NO_BODY = 0,
        HET_LENGTH,
        HET_CHUNKED,
        HET_EOF
    };
    enum HTTPChunkedState {
        CHUNK_SIZE = 0,
        CHUNK_DATA,
        CHUNK_DATA_CRLF,
        TRAILER
    };
    virtual void reset();
    void setErrorNo(int errorNo);

public:
    HTTPStreamingStep _step;
    HTTPChunkedState _chunkState;
    HTTPEncodingType _encodingType;
    size_t _dataLength;
    size_t _drainedLength;
    size_t _headersCount;
};

} /*end namespace anet*/
