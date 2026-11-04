#pragma once
/**
 *@file LoggingEvent.h
 *@brief the file to declare LoggingEvent struct.
 *
 *@version 1.0.4
 *@date 2009.03.05
 *@author bingbing.yang
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/syscall.h>
#include <sys/time.h>
#include <syscall.h>
#include <time.h>
#include <unistd.h>

namespace alog {

// Log level names array - defined in LoggingEvent.cpp
extern const char *s_gLevelName[];
static constexpr std::size_t s_gFmtLength = 27;
static constexpr std::size_t s_gFmtZoneLength = 33;

extern int32_t sysPid;
extern thread_local int64_t sysTid;

struct LoggingEvent {
public:
    /**
     *@brief to get formatted current time stored in param cur as a return.
     *
     *@param cur formatted time as a return value.
     *@param length the cur buffer's length
     */
    void getCurTime(char cur[], int length, char zoneTime[], int zoneTimeLength) {
        struct timeval t;
        gettimeofday(&t, NULL);
        struct tm tim;
        ::localtime_r(&t.tv_sec, &tim);
        snprintf(cur,
                 length,
                 "%04d-%02d-%02d %02d:%02d:%02d.%06ld",
                 tim.tm_year + 1900,
                 tim.tm_mon + 1,
                 tim.tm_mday,
                 tim.tm_hour,
                 tim.tm_min,
                 tim.tm_sec,
                 t.tv_usec);
        strcpy(zoneTime, cur);
        char *p = zoneTime + 26;
        strftime(p, zoneTimeLength - 26, ":%z", &tim);
    }

    /**
     *@brief constructor.
     *
     *@param name logger name.
     *@param msg user's logger message.
     *@param level logging level.
     */
    LoggingEvent(const std::string &name,
                 std::string msg,
                 const uint32_t level,
                 const std::string &file = "",
                 const int &line = 0,
                 const std::string &func = "")
        : loggerName(name)
        , message(std::move(msg))
        , level(level)
        , levelStr(s_gLevelName[level])
        , file(file)
        , line(line)
        , function(func)
        , pid(sysPid)
        , tid(sysTid) {
        getCurTime(loggingTime, s_gFmtLength, loggingZoneTime, s_gFmtZoneLength);
    }

    /** The logger name. */
    std::string loggerName;
    /** The application supplied message of logging event. */
    const std::string message;
    /** Level of logging event. */
    const uint32_t level;
    /** Level string of logging event. */
    const std::string levelStr;
    /** File name. */
    const std::string file;
    /** Line count. */
    const int line;
    /** Function name. */
    const std::string function;
    /** Process id */
    const int pid;
    /** Thread id */
    const long tid;
    /** Time yyyy-mm-dd hh:mm:ss:ms*/
    char loggingTime[s_gFmtLength];
    /** Time yyyy-mm-dd hh:mm:ss:ms*/
    char loggingZoneTime[s_gFmtZoneLength];
};
} // namespace alog
