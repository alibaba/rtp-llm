#pragma once
/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-07-17 14:38
 * Author Name: beifei
 * Author Email: beifei@taobao.com
 */

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>

#include "kmonitor/client/common/Common.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

class Util {
public:
    static bool GetHostIP(std::string &host);
    static int64_t TimeAlign(int64_t curTime, int64_t timeBucket);
    static bool HostnameToIp(std::string hostname, std::string &ip);
    static const std::string GetMachineName();
    static bool UseIPV6();
    static bool IsIPV6(const std::string &ip);
    static std::string GetParentHostIPV6();
    static bool IsIPV4(const std::string &ip);
    static std::string GetParentHostIPV4();

private:
    Util(const Util &);
    Util &operator=(const Util &);

    static const std::string getMachineName(std::istream &is);
    static std::string getHostLine(std::istream &is, std::function<bool(const std::string &)> func);

private:
    static const int BUF_SIZE = 1000;
    static const std::string HOST_INFO_PATH;
    static const std::string HOST_INFO_IPV6_PATH;
};

END_KMONITOR_NAMESPACE(kmonitor);
