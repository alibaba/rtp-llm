#pragma once
/**
 * File name: aneterror.h
 * Author: zhangli
 * Create time: 2008-12-21 15:29:40
 * $Id$
 *
 * Description: ***add description here***
 *
 */

//*****add include headers here...
namespace anet {
class AnetError {
public:
    static const int INVALID_DATA;
    static const char *INVALID_DATA_S;

    static const int CONNECTION_CLOSED;
    static const char *CONNECTION_CLOSED_S;

    static const int PKG_TOO_LARGE;
    static const char *PKG_TOO_LARGE_S;

    /**
     * HTTP related error no
     */
    static const int LENGTH_REQUIRED;
    static const char *LENGTH_REQUIRED_S;

    static const int URI_TOO_LARGE;
    static const char *URI_TOO_LARGE_S;

    static const int VERSION_NOT_SUPPORT;
    static const char *VERSION_NOT_SUPPORT_S;

    static const int TOO_MANY_HEADERS;
    static const char *TOO_MANY_HEADERS_S;
};

} /*end namespace anet*/
