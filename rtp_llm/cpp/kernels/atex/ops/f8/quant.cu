/*
template<uint32_t TPB, uint32_t VPT>
__global__ void device_minmax_pertensor_quant_f16_fp8e4m3(
    const fp16x2_t* const x,
    const uint32_t numel,
    uint32_t *workspace,
    fp8_t *y, fp32_t *scale
) {
    // 统计 x 函数中的最大值，并以最大值的 448 分之一计算scale，从而对给定的 tensor 进行 fp8 量化
    // 量化后的值存入 y, 量化 scale 存入 scale，这个算子需要提前传入一个值为 0 的 workspace

    for (uint32_t i = 0; i < )
}
    */
