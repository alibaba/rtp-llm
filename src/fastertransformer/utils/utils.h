
#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define T_SWITCH(COND, V, A, B, ...)            \
    [&] {                                       \
        if (COND) {                             \
            using V = A;                        \
            return __VA_ARGS__();               \
        } else {                                \
            using V = B;                        \
            return __VA_ARGS__();               \
        }                                       \
    }()

#define V_SWITCH(COND, V, A, B, ...)            \
    [&] {                                       \
        if (COND) {                             \
            constexpr auto V = A;               \
            return __VA_ARGS__();               \
        } else {                                \
            constexpr auto V = B;               \
            return __VA_ARGS__();               \
        }                                       \
    }()
