#pragma once
/**
 *@file Configurator.h
 *@brief the file to declare Configurator class.
 *
 *@version 1.0.0
 *@date 2008.12.22
 *@author Bingbing Yang
 */

#include <map>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

#include "Logger.h"

namespace alog {
class Properties;
class Appender;

/**
 *@class Configurator
 *@brief the class to configure loggers by file and rootLogger basicly.
 *
 *@version 1.0.0
 *@date 2008.12.22
 *@author Bingbing Yang
 *@warning
 */
class Configurator {
public:
    typedef std::map<std::string, Appender *> AppenderMap;
    /**
     * @brief configure loggers by file.
     * @param conf file path.
     **/
    static void configureLogger(const char *initFileName) noexcept(false);

    /**
     * @brief configure loggers by string.
     * @param conf file content.
     **/
    static void configureLoggerFromString(const char *fileContent) noexcept(false);
    /**
     * @brief configure root logger basicly.
     **/
    static void configureRootLogger();

    /**
     * @brief translate delaytime str to seconds.
     * @param delaytime string.
     * if str endwith 's', delaytime is in second,
     * if str endwith 'm', delaytime is in minute,
     * if str endwith 'h', delaytime is in hour,
     * if str endwith 'd', delaytime is in day.
     * otherwith delaytime is in hour.
     */
    static uint32_t translateDelayTime(const std::string &str);

private:
    static void innerConfigureLogger(std::istream &is);
    static void globalInit(Properties &properties);
    static void initLoggers(std::vector<std::string> &loggers, Properties &properties);
    static void
    configureLogger(const std::string &loggerName, AppenderMap &allAppenders, Properties &properties) noexcept(false);
    static void initAllAppenders(AppenderMap &allAppenders, Properties &properties) noexcept(false);
    static Appender *instantiateAppender(const std::string &name, Properties &properties) noexcept(false);
    static void setLayout(Appender *appender, const std::string &appenderName, Properties &properties);
    static uint32_t getLevelByString(const std::string &levelStr);
    static std::string transformPattern(const std::string &inputPattern);
    static std::string getProgressName();
    static std::string getLastDirectory(uint32_t layers);
    static std::string getHomeDirectory();
};
} // namespace alog
