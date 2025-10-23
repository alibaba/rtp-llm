struct uint8x4_t {
    uint8_t x,y,z,w;
}

void thread_select_42(
    uint8x4_t *x,
    uint8x4_t *y,
    uint8_t *meta
) {
    __perm(x[0], 0, )
}