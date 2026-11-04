#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httpstreamingcontext.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/aneterror.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/streamingcontext.h"

namespace anet {
HTTPStreamingContext::HTTPStreamingContext() {
    _step = HSS_START_LINE;
    _dataLength = 0;
    _drainedLength = 0;
    _headersCount = 0;
    _chunkState = CHUNK_SIZE;
}

HTTPStreamingContext::~HTTPStreamingContext() {}

void HTTPStreamingContext::reset() {
    _step = HSS_START_LINE;
    _dataLength = 0;
    _drainedLength = 0;
    _headersCount = 0;
    _chunkState = CHUNK_SIZE;
    StreamingContext::reset();
}

void HTTPStreamingContext::setErrorNo(int errorNo) {
    setBroken(true);
    _errorNo = errorNo;
    if (errorNo == AnetError::INVALID_DATA) {
        _errorString = AnetError::INVALID_DATA_S;
    } else if (errorNo == AnetError::PKG_TOO_LARGE) {
        _errorString = AnetError::PKG_TOO_LARGE_S;
    } else if (errorNo == AnetError::LENGTH_REQUIRED) {
        _errorString = AnetError::LENGTH_REQUIRED_S;
    } else if (errorNo == AnetError::URI_TOO_LARGE) {
        _errorString = AnetError::URI_TOO_LARGE_S;
    } else if (errorNo == AnetError::VERSION_NOT_SUPPORT) {
        _errorString = AnetError::VERSION_NOT_SUPPORT_S;
    } else if (errorNo == AnetError::TOO_MANY_HEADERS) {
        _errorString = AnetError::TOO_MANY_HEADERS_S;
    } else if (errorNo == AnetError::CONNECTION_CLOSED) {
        _errorString = AnetError::CONNECTION_CLOSED_S;
    } else {
        _errorString = NULL;
    }
}

} /*end namespace anet*/
