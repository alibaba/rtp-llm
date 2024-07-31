#include "alog/Appender.h"
#include "alog/Logger.h"
#include "autil/Log.h"

namespace rtp_llm {
    class AccessLogger {
    private:
        AccessLogger() {
            logger_ = alog::Logger::getLogger("rtp_llm.logutil");
            if (logger_ == nullptr) {
                throw std::runtime_error("should not be nullptr");
            }
            alog::Logger::MAX_MESSAGE_LENGTH = 102400;
            appender_ = (alog::FileAppender*)alog::FileAppender::getAppender("logs/rpc_access.log");
            appender_->setCacheLimit(1024);
            appender_->setHistoryLogKeepCount(5);
            appender_->setAsyncFlush(false);
            appender_->setFlushIntervalInMS(1000);
            appender_->setFlushThreshold(1000);

            logger_->setAppender(appender_);
            logger_->setInheritFlag(false);
            logger_->setLevel(alog::LOG_LEVEL_INFO);
        }
    protected:
        alog::Logger *logger_;
        alog::FileAppender* appender_;
    public:
        static const AccessLogger* getAccessLogger() {
            static AccessLogger access_logger_;
            return &access_logger_;
        }
        void recordSucess(std::string request, std::string response) const {
            logger_->log(alog::LOG_LEVEL_INFO,"requset: %s, response: %s", request.c_str(), response.c_str());
        }
        void recordFail(std::string request, std::string exception_msg) const {            
            logger_->log(alog::LOG_LEVEL_ERROR,"requset: %s, exception_msg: %s", request.c_str(), exception_msg.c_str());
        }
};
}
