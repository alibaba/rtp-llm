#pragma once
/**
 *@file Properties.h
 *@brief the file to declare Properties class.
 *
 *@version 1.0.0
 *@date 2008.12.22
 *@author Bingbing Yang
 */

#include <map>
#include <string>

namespace alog {

/**
 *@class Properties
 *@brief the class to store configure file's properties.
 *
 *@version 1.0.0
 *@date 2008.12.22
 *@author Bingbing Yang
 *@warning
 */
class Properties : public std::map<std::string, std::string> {
public:
    Properties();
    ~Properties();

    void load(std::istream &in);
    // void save(std::ostream& out);

    int getInt(const std::string &property, int defaultValue);
    bool getBool(const std::string &property, bool defaultValue);
    std::string getString(const std::string &property, const char *defaultValue);

private:
    void _substituteVariables(std::string &value);
};
} // namespace alog
