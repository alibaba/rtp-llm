import os
import threading

from .util import naviSchedModule


NAVI_SCHED_ALOG_CONFIG = r"""alog.rootLogger=INFO, NaviSchedAlogAppender
alog.max_msg_len=2000000
alog.appender.NaviSchedAlogAppender=FileAppender
alog.appender.NaviSchedAlogAppender.fileName=./logs/navi_sched_alog.log
alog.appender.NaviSchedAlogAppender.layout=PatternLayout
alog.appender.NaviSchedAlogAppender.layout.LogPattern=[%%d] [%%l] [%%t,%%F -- %%f():%%n] [%%m]
alog.appender.NaviSchedAlogAppender.cache_limit=128
alog.appender.NaviSchedAlogAppender.compress=true
alog.appender.NaviSchedAlogAppender.max_file_size=1024
alog.appender.NaviSchedAlogAppender.log_keep_count=20
alog.appender.NaviSchedAlogAppender.async_flush=true
alog.appender.NaviSchedAlogAppender.flush_threshold=1024
alog.appender.NaviSchedAlogAppender.flush_interval=1000

alog.logger.multi_call=INFO, GigAppender
inherit.multi_call=false
alog.appender.GigAppender=FileAppender
alog.appender.GigAppender.fileName=./logs/gig.log
alog.appender.GigAppender.layout=PatternLayout
alog.appender.GigAppender.layout.LogPattern=[%%d] [%%l] [%%t,%%F -- %%f():%%n] [%%m]
alog.appender.GigAppender.compress=true
alog.appender.GigAppender.max_file_size=1024
alog.appender.GigAppender.log_keep_count=20
alog.appender.GigAppender.async_flush=true
alog.appender.GigAppender.flush_threshold=1
alog.appender.GigAppender.flush_interval=1000
"""

_alog_lock = threading.Lock()
_alog_configured = False


def configureNaviSchedAlog() -> None:
    global _alog_configured
    if _alog_configured:
        return
    with _alog_lock:
        if _alog_configured:
            return
        os.makedirs("./logs", exist_ok=True)
        naviSchedModule()._configure_alog(NAVI_SCHED_ALOG_CONFIG)
        _alog_configured = True
