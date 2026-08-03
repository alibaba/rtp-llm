# BlockTree分支合入P2P适配说明

本次合并以`dev/dsv4_yichen_block_tree_cache`为基准，保留BlockTree的分配、本地复用和释放逻辑，P2P只负责在PD分离场景补齐未本地命中的KV。

## 调用链

Decode首先按BlockTree原路径执行match和malloc，得到`local_reuse_len`及目标block。存在P2PConnector时，StreamCacheResource同时创建P2P asyncLoad；BlockTree和P2P两个context均完成后才继续执行。P2P context一旦创建就是必须完成的PD握手：部分命中时补齐KV，全命中时仍需通过no-transfer返回首token并释放Prefill资源，任一P2P失败都上报。

## 关键关联点

1.`KVCacheResource`仍由BlockTree分配并持有。P2P使用`BlockRefType::Connector`增加传输期引用，完成、取消或超时后释放，不替代Tree自身引用。
2.P2P读取范围由Decode端BlockTree的实际本地复用结果决定。Prefill不再接收`start_block_index`，而是按Topology和分组策略提供该请求可传输的KV；Decode只为未被Tree覆盖的目标block注册接收，来源多提供的key可忽略。
3.P/D两端共用`CacheTopology`解释layer、tag及FULL/SWA/LINEAR分组。Decode用Topology将Tree分配的block转成接收buffer；Prefill逐层计算完成后，用全局layer_id和tag提供对应buffer。
4.`MetaImpl`保留，用于把Stream的request_id、unique_key、deadline、Prefill地址和TP规模传给P2P，避免恢复Coordinator调用链。Coordinator源码不删除，Tree和其他Connector的现有行为不变。
5.100%Tree命中时，Decode仍发起`StartLoad(no_transfer)`。Prefill不传KV，但会返回首token和复用信息，保证PD请求正常收尾。

## 当前边界

非CP路径需覆盖Tree无命中、部分命中和全命中三类场景。CP/CSA下的rank投影、每rank期待集合及ready-empty语义仍是遗留项，按`block_tree_p2p_merge_issues.md`继续处理。
