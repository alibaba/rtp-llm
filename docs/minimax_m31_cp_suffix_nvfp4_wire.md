# MiniMax-M3.1 CP4 NVFP4 suffix wire R16（实验，默认关闭）

仅same-layer CP4 native路径启用 `RTP_LLM_CP_SUFFIX_NVFP4_WIRE=1`；默认0。固定所有rank一致，workspace生命周期不切format。本轮只验收Zero-CTA `NCCL_CTA_POLICY=2`；ordinary完整链顺序测试结果不稳定，不作为性能推荐。

数据流：Gemma norm/RoPE先输出与旧路径相同BF16-rounded carrier；`nvfp4_cp_wire.py` 的sender在main按原amax/clamp/E4M3/E2M1-RNE量化为uint8[R,648]，side等待producer后做一次AG；`nvfp4_cp_wire.py` receiver R16（原 multirow 模块保留兼容 import）将原始codes/scales按unpad、working slots及全长persistent owner-map复制到六planes。工作main K/V为FI common 73728-byte page，显式FI selector；index与persistent仍MMA ABI，PD传输ABI不变。

每row9×72B，原BF16为2304B，实测payload减少71.875%。prefix两AG保持不变；restore完成、suffix wait、scatter、tail-clear、retire顺序保留。固定streams、registered symmetric buffers、generation、跨slot fences及owner存活保持；quant-pack后释放本地BF16carrier。默认OFF fallback与冻结FI writer/tail调用相同，不使用单行prototype scatter。

R16单GPU byte/Graph/owner/padding/>2GiB-stride/isolated invalid-source断言通过，真实CP4生产SIDE顺序CE正反序均改善；ordinary不稳定。完整模型 `final_wire_cp4` 五场景20轮960RPC零错误，四rank实际每层60次pack/scatter，suffix peer bytes437944320→123171840。Pure40/60模型耗时1.692255/2.489655s，相对FI下降2.461%/2.454%。四rank restore与suffix peer-copy交叠仍0，不能称新AG overlap。

精度未签收：GSM436/500相同，LongBench66.8171467→66.2377381，4/160输出不同，其中qasper yes→no贡献−0.625 raw分；5个截断IDs相同。需固定该请求，先捕捉首token logits和最早不同内部状态，再作原因归属；不根据单次总分/全文等同比较宣称算子精度回归或完全等价。因此保持实验默认OFF，不用于默认生产配置。

完整配置、内存、TPM/TPMS、四rank timeline、源码/native identity与失败保留见 `/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/final_wire_cp4/RESULT.md` 和主分析文档。主树接入preserves Decode/C++ WIP；runtime模型包冻结，主树MSA另保留既有两处Decode阈值160，不把当前HEAD等同本轮完整运行包。
