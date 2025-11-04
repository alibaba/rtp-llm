#progma once

namespace atex {

template<int VPT>
struct BytesToType;

template<>
struct BytesToType<2> {
    using type = uint16_t;
};
template<>
struct BytesToType<4> {
    using type = uint32_t;
};
template<>
struct BytesToType<8> {
    using type = uint64_t;
};
template<>
struct BytesToType<16> {
    using type = float4;
};

template<int Bytes>
__device__ inline void copy(const void* local, void* data) {
    using T = typename BytesToType<Bytes>::type;

    const T* in  = static_cast<const T*>(local);
    T*       out = static_cast<T*>(data);
    *out         = *in;
}

};  // namespace atex