#include "maga_transformer/cpp/common/fatal_util.h"
#include "src/fastertransformer/utils/logger.h"

#define RETURN_IF_STATUS_OR_ERROR(status_or)                                                                           \
    do {                                                                                                               \
        auto&& _status_or = (status_or);                                                                               \
        if (ABSL_PREDICT_FALSE(!_status_or.ok()))                                                                      \
            return _status_or.status();                                                                                \
    } while (0)

#define RETURN_IF_STATUS_ERROR(status)                                                                                 \
    do {                                                                                                               \
        ::absl::Status _status = (status);                                                                             \
        if (ABSL_PREDICT_FALSE(!_status.ok()))                                                                         \
            return _status;                                                                                            \
    } while (0)

#define THROW_IF_STATUS_ERROR(status)                                                                                  \
    do {                                                                                                               \
        ::absl::Status _status = (status);                                                                             \
        if (ABSL_PREDICT_FALSE(!_status.ok()))                                                                         \
            RAISE_FATAL_ERROR(_status.ToString());                                                                     \
    } while (0)
