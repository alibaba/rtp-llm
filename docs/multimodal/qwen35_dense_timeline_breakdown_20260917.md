# 当前 dense FA4 timeline 的 kernel 与调度耗时

来源为删除 QK 布局拷贝后的 engine.nsys-rep。固定 B16：单请求含16个视频、27层、4次稳态均值；C16：64个独立请求，实际合批为4批B4和4批B12。前两个C16 batch不在Nsight捕获窗口内，但host/CUDA event记录完整；全部4次固定B16完整捕获。

## 固定 B16：kernel 按用途分类

每行是该用途下实际CUDA kernel的执行时长求和，均为每批平均，不是CPU API时长。kernel调用数可多于逻辑算子次数；完整符号和单次调用均值见CSV。

| 用途 | CUDA kernel调用数/批 | kernel ms/批 |
|---|---:|---:|
| fa4 | 27 | 594.292 |
| mlp_fc2 | 27 | 254.027 |
| layer_norm | 54 | 244.092 |
| mlp_fc1 | 27 | 206.029 |
| mlp_gelu | 216 | 182.303 |
| qkv_projection | 27 | 169.245 |
| rope | 27 | 153.056 |
| residual | 108 | 70.760 |
| output_projection | 27 | 68.581 |
| gpu_resize | 80 | 65.371 |
| gpu_normalize_fold | 64 | 23.647 |
| merger | 5 | 21.424 |
| nvdec_cuda | 3744 | 10.239 |
| patch_embedding | 1 | 2.973 |
| production_feature_hash | 16 | 2.451 |
| frame_token_assembly | 16 | 2.378 |
| position_interpolation | 17 | 1.886 |
| gpu_preprocess_misc | 16 | 1.722 |
| nv12_to_rgb | 16 | 1.392 |
| vision_misc | 2 | 1.284 |
| batch_cat_text_copy | 1 | 1.165 |
| rotary_metadata | 2 | 0.325 |
| attention_layout | 0 | 0.000 |

其中ViT：kernel sum=1970.276ms，copy sum=1.201ms，kernel/copy interval union=1971.443ms，GPU activity envelope=1972.436ms，未覆盖间隙=0.993ms；CUDA event=1972.448ms。sum与union不同，不能把重叠项相加。

## 固定 B16：外层耗时与调度

| 项目 | 平均ms/批 | 解释 |
|---|---:|---|
| NVDEC+GPU预处理 CUDA event | 393.438 | 包含解码等待、转换、resize、归一化等 |
| ViT CUDA event | 1972.448 | 模型设备可见耗时 |
| frame/token assembly CUDA event | 10.341 | 包含kernel、copy及等待 |
| 三个大阶段外的batch CUDA event时间 | 9.372 | 主要包含输入拼接、文本embedding H2D及间隙，非纯CPU调度 |
| batched_embedding CUDA event | 2385.598 | 上述四项之和 |
| MMScheduler._execute_batch wall | 2389.799 | 完整批次服务时间，包含生产feature hash及结果发布 |
| batched_embedding外的host span差值 | 4.210 | feature hash 3.476ms，其余包装/结果发布等0.734ms；均为外层墙钟差值 |
| CPU vision_forward调用span | 35.518 | Python/框架及提交kernel，异步排队；与GPU工作重叠，不再加到GPU耗时上 |
| kernel launch API host interval union | 2.975 | 属于上一行的子集，按时间区间去重 |
| ViT内GPU activity未覆盖间隙 | 0.993 | 非CPU耗时测量值，也不是整卡GPU空闲率 |

CUDA memcpy显式记录（每批）：文本embedding H2D 24,117,248bytes，GPU拷贝均值6.463ms（四次2.212/19.265/2.278/2.097ms）；NVDEC辅助D2D合计19.403ms；归一化/折叠D2D合计3.613ms；ViT内copy合计1.201ms。这些已包含在外层阶段内。NVDEC辅助CUDA kernel合计10.239ms不代表硬件视频解码总耗时；16路NVDEC worker的host区间union约305.371ms。

最大的cudaMemcpyAsync CPU API span约2018.166ms/批；其时间区间完整覆盖了ViT GPU envelope。该API墙钟包含等待前面GPU工作的时间，实际关联的文本H2D GPU memcpy仅均值6.463ms，不能把约2秒当作额外的拷贝或调度成本。

## 实际 C16：每请求平均等待与服务时间

| 项目 | 平均ms/请求 |
|---|---:|
| 入口至提交（包含文件/metadata准备） | 58.043 |
| 提交后排队/等待合批 | 776.945 |
| 所在batch完整服务时间 | 1500.706 |
| batch结束至请求返回 | 6.440 |
| 请求总RT | 2342.135 |

输入准备中，文件读取均值2.507ms、metadata处理34.183ms；这些是input 58.043ms的子阶段，不能再次相加。
相邻C16批次在host上的间隔平均12.000ms，范围10.188～17.460ms，配置合批等待窗口10ms。该间隔包含合批等待及调度/唤醒，未单独测得线程纯CPU运行时间。
排队776.945ms指从提交到本批启动，包括等前批次GPU工作完成，并非CPU执行调度逻辑776.945ms。

以上均为当前带profiler诊断的测量值；不是无profiler吞吐基准RT，也不包含gRPC序列化、网络或LLM。

- [完整33组kernel符号明细](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/diagnostic/current-kernel-breakdown.csv)
- [结构化耗时数据](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/diagnostic/current-timeline-breakdown.json)
- [memcpy等待关联证据](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/diagnostic/memcpy-host-wait-evidence.json)
- [原始时间线](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/diagnostic/engine.nsys-rep)
