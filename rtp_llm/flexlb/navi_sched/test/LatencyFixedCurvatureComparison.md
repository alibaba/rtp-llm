# MI308 固定曲率与递增凹函数离线对照

## 方法和口径

- 样本：A=`/tmp/navi-rate-config-aware-0106`，B=`/tmp/navi-independent-combined-original-0106`，各 10 个 MI308 节点。每节点分别拟合，不把不同节点的参数混在一起。
- 公式：线上凸式 `u+sqrt(u²+K)`；固定 K 时只移除 K 的优化维度，其余参数全部重新拟合。凹式对照为 `u-c*sqrt(u²+K)`，`c=0.5/0.9`，其导数始终大于 `1-c`，所以单调递增。
- **K 不会在当前公式里自动抵消。** 记 `h(z)=z+sqrt(z²+1)`，则非线性贡献为 `A*sqrt(K)*[h(u/sqrt(K))-h(u0/sqrt(K))]`。把内层系数同比缩放以保持 `u/sqrt(K)` 不变时，只有再把外层 `A` 反向缩放才能保证非线性贡献恒等；本实验及日志中的 `A=1500` 固定。源码的 `reparameterize` 对内层系数和 K 使用了这种缩放，但它还缩放了非线性项外的 p15，不能据此推断整个模型在修改 A 时严格恒等。一个 MI 节点从 K=894.79 变到 100、内层同比缩放后，固定 A 的预测贡献中位绝对变化为 2132 ms；A 反向缩放后非线性贡献最大差异小于 `2e-11 ms`。因此实验看到 K 的影响符合当前公式，且 K 不只是一个独立的“曲率旋钮”。
- 损失：训练样本的 `sum((log1p(pred_ms)-log1p(actual_ms))²)`；不按压力加权。留出集不参加参数拟合。`blocks` 按 120 秒块每 4 块留 1 块；`forward` 留每节点最后 25% 的时间样本。
- 低压：`running<=4 && waiting<=4`；高压：`waiting>=33`。高压上升段是本 30 秒块的平均 waiting 比前一块至少多 4 的高压样本，仅为压力上升代理，不代表逐查询严格上升。
- 表中单位均为 ms，`MAE` 为绝对误差平均值；`偏差=actual-pred`，正值表示低估。表里的最优固定 K 是事后查看留出集选出的探索性结果，**不可把它当无偏的选参评估**。

## 原凸式与固定 K

|窗口/划分|方案|全部 MAE|低压 MAE|高压 MAE|高压上升 MAE|高压上升偏差|
|---|---:|---:|---:|---:|---:|---:|
|A/blocks|自由 K|4760|230|13869|14940|+3334|
|A/blocks|固定 K=100|4638|192|13540|14804|+3030|
|B/blocks|自由 K|5256|64|12143|16784|+751|
|B/blocks|固定 K=10000|5099|63|11779|16515|+644|
|A/forward|自由 K|6772|104|12120|14236|+1478|
|A/forward|固定 K=1|6575|108|11707|13645|+813|
|B/forward|自由 K|7328|61|11536|17259|+750|
|B/forward|固定 K=10000|7251|61|11420|17280|+696|

固定 K 有约 1%-3% 的全体 MAE 改善，但两个窗口/两种留出划分选择的 K 不一致，高压上升段仍有明显绝对误差。高压逐节点 p95 绝对误差的中位数：A/blocks 自由 K 57322、K=100 为 59297（恶化）；B/blocks 自由 K 29485、K=10000 为 27699（改善）。低压短请求（真实耗时<256ms）在 A/blocks 的 MAE 为自由 K 33、K=100 为 34；B/blocks 为自由 K 46、K=10000 为 45。

扩展网格到 K=0.01/0.05/0.1 后，A/blocks 的自由 K 多初值拟合有一个节点收敛到 K=0.000309：训练损失由固定 K=1 的 176.15 降为 166.77，但该节点留出低压 MAE 从 437 暴涨到 65229。10 节点汇总留出低压 MAE 从 189 升至 6390。这是自由 K 极小值分支造成的**静态低压泛化风险**，不是已证实的在线零点漂移；严格零输入时非线性项相减为零，零点参数仍是 p7。固定 K 避开这个分支，但不能仅凭此指定线上 K。

## 递增凹函数对照

|窗口/划分|方案|全部 MAE|低压 MAE|高压 MAE|高压上升 MAE|高压上升偏差|
|---|---:|---:|---:|---:|---:|---:|
|A/blocks|凸式固定 K=100|4638|192|13540|14804|+3030|
|A/blocks|凹式 c=0.5,K=10000|4857|270|14094|14800|+5153|
|B/blocks|凸式固定 K=10000|5099|63|11779|16515|+644|
|B/blocks|凹式 c=0.5,K=1|5366|66|12414|16774|+1711|
|A/forward|凸式固定 K=1|6575|108|11707|13645|+813|
|A/forward|凹式 c=0.9,K=10000|6716|109|11993|13313|+2240|
|B/forward|凸式固定 K=10000|7251|61|11420|17280|+696|
|B/forward|凹式 c=0.9,K=1|7462|64|11757|17179|+1877|

凹式有时降低上升段 MAE，但有符号低估明显变大，整体高压也未跨窗口改善；目前数据不支持直接改成递增凹函数。内层系数重拟合可部分补偿 K 的改变，但外层系数固定时不存在精确抵消；单看拟合后 K 的数值不能识别真实物理曲率。固定 K 降低的是拟合自由度；是否改善在线跟随，必须在同一在线回放状态机中冻结 K 并重播更新过程，本脚本没有做这一步。

## 复现

```bash
OPENBLAS_NUM_THREADS=1 /home/zq103303/vllm/.venv-navi-sched/bin/python aios/navi_sched/test/CompareLatencyFixedCurvature.py /tmp/navi-rate-config-aware-0106 --workers 10 --max-nfev 800 --curvatures 0.25 1 100 900 3000 10000 > /tmp/navi-fixed-k-A-final.json
OPENBLAS_NUM_THREADS=1 /home/zq103303/vllm/.venv-navi-sched/bin/python aios/navi_sched/test/CompareLatencyFixedCurvature.py /tmp/navi-independent-combined-original-0106 --workers 10 --max-nfev 800 --curvatures 0.25 1 100 900 3000 10000 > /tmp/navi-fixed-k-B-final.json
OPENBLAS_NUM_THREADS=1 /home/zq103303/vllm/.venv-navi-sched/bin/python -m unittest CompareLatencyFixedCurvatureTest.py
```

严格时间末段留出用 `--split-mode forward`；递增凹式用 `--shape concave_05` 或 `--shape concave_09`。完整逐节点误差在对应 JSON 的 `hosts[].experiments` 中。
