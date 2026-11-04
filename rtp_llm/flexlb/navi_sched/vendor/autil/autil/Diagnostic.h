#pragma once
#define PRAGMA(TXT) _Pragma(#TXT)
#ifdef __clang__
#define DIAGNOSTIC_PUSH _Pragma("clang diagnostic push")
#define DIAGNOSTIC_POP _Pragma("clang diagnostic pop")
#define DIAGNOSTIC_IGNORE(XXX) PRAGMA(clang diagnostic ignored XXX)
#elif defined(__GNUC__)
#define DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#define DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")
#define DIAGNOSTIC_IGNORE(XXX) PRAGMA(GCC diagnostic ignored XXX)
#endif
