# Block Tree 与 P2P 合并 Review 记录

本文按 review 轮次记录 P2 及以上问题、修复方案和验证状态。每个问题保持简短；CP/CSA 的集中遗留见 `block_tree_p2p_merge_issues.md`。

## 第一轮：调用链与资源所有权

### 1.P2P仍经过Coordinator

问题：P2P读写仍复用Coordinator轮询、引用计数和构建依赖，使新Tree链路存在两套调度与资源所有权。

修复：Decode加载直接由KVCacheManager调用P2PConnector；逐层写入直接调用P2P接口；引用统一使用Connector类型；仅移除P2P路径上的Coordinator调用和构建依赖，保留Coordinator源码及原有目标。

### 2.P2P读取在Engine线程发起慢操作

问题：asyncRead创建连接及RPC会阻塞Engine调用线程，行为与其他异步Connector不一致。

修复：asyncRead仅创建并登记上下文，RPC启动下沉到P2P自己的线程池；checker负责推进完成、取消与资源释放。

### 3.Stream侧错误重试已消费资源

问题：StartLoad已取走Prefill资源后，Stream重试无法再次获得同一资源，且可能重复启动物理传输。

修复：移除P2P Stream重试，失败直接上报，与Tree路径的单次资源消费语义一致。

## 第二轮：Tree适配与全命中

### 4.Tree全命中缺少Prefill完成握手

问题：Decode本地100%命中时跳过P2P加载，Prefill无法收到完成信号，整请求资源和side-channel可能滞留。

修复：新增no_transfer握手。Decode仍发StartLoad和TP广播，但不注册接收buffer、不发起物理传输；Prefill完成请求终态和side-channel返回。

### 5.逐层写入未使用Tree拓扑

问题：P2P沿用单一group映射，FULL、SWA、LINEAR及多tag场景可能生成与Decode不一致的layer/tag和key。

修复：Prefill和Decode统一使用CacheTopology与buildCacheStorePlan；逐层回调按全局layer和tag登记，Decode按相同规则构造目标范围。

### 6.Meta实现被裁剪

问题：临时StreamConnectorMeta替代了原MetaImpl，丢失原接口形态及后续扩展承载点。

修复：恢复MetaImpl，在其中补充P2P routing context和GenerateStream引用，保留原Meta语义。

## 第三轮：传输正确性与错误传播

### 7.Decode目标覆盖校验不完整

问题：TCP/RDMA允许目标key或子block未被来源覆盖时仅记录日志，可能把部分KV写入判定为成功。

修复：每个Decode目标key及其全部子block必须存在且尺寸匹配，否则返回BUFFER_MISMATCH；Prefill多提供的block允许忽略。

### 8.逐层回调分发失败未上报

问题：P2P逐层写入失败仅记日志，Prefill仍等待缺失的topology项，最终表现为长超时。

修复：writeByLayerTag返回bool，后台写线程保存异常，模型forward收口时重新抛出并上报请求失败。

### 9.非CP空层或非法key被静默跳过

问题：发送计划非空但block全部无效，或cache key不能转成整数时，代码直接continue，Prefill会等待对应layer/tag直到传输deadline。

修复：非CP路径立即抛错并传回模型层；CP空分片暂保持现状，纳入CP遗留。

## 第四轮：期限与终态

### 10.固定一小时tombstone

问题：已完成请求终态固定保留一小时，与用户请求deadline脱节，可能占用过多状态或过早失效。

修复：StartLoad同时携带请求deadline和物理传输deadline；终态保留到原请求deadline，非法旧值才使用配置回退。

### 11.side-channel沿用旧资源deadline

问题：资源被StartLoad取走后，后到的side-channel通知可能回退到旧的资源持有期限，早于或远晚于实际传输结束。

修复：消费资源时登记活动传输deadline；后续side-channel统一复用该值，完成或清理时同步移除。

### 12.逐层buffer未切换到传输deadline

问题：首层回调先固定资源等待上界；若Decode在等待窗口后段发起StartLoad，合法传输开始后buffer仍可能按旧上界删除。

修复：区分首次登记和StartLoad激活。逐层回调不能滚动上界；StartLoad可将固定上界提升到本次传输deadline，等待GPU event的上下文同步读取该值。

## 第五轮：异步生命周期

### 13.Broadcast部分下发后丢失所有权

问题：前几个rank已启动RPC、后续rank创建reader失败时返回nullptr，已启动调用的context和CQ可能提前析构。

修复：失败结果继续持有全部context，未启动rank直接标记失败；析构先取消并限时drain，超时后交给后台drainer持有至CQ关闭。

### 14.Decode目标block提前释放

问题：READ回调未确认物理操作停止时，AsyncContext已完成并释放目标block，旧RDMA可能继续写入被复用的block。

修复：失败后在配置上界内保留Decode目标引用，轮询各rank的sealed lease；全部停止后提前释放，到达固定上界则强制释放并记录错误。

### 15.Prefill非主rank资源未释放

问题：StartLoad只取走主rank整请求资源，其他Prefill TP rank在广播发送结束后仍持有Connector引用直到超时。

修复：非主rank在HANDLE_READ或no_transfer处理完成后主动标记终态并清理本地资源；主rank在side-channel消费完成后统一终结。

### 16.Decode取消暂存固定一小时

问题：CANCEL_READ先于READ注册时，worker把取消key固定保留一小时。故障风暴下会持续扩大暂存表，并与请求真实生命周期脱节。

修复：Decode异步上下文把原请求deadline带入取消广播；worker将暂存终态保留到该deadline。只有旧调用未提供deadline时才使用配置回退。

## 第六轮：独立P2P输入边界

### 17.逐层写入缺少输入形状校验

问题：P2P可在不创建CacheStore时独立逐层写入；若cache key数量不能按batch切分、block table行数不足或长度tensor过短，原逻辑可能静默生成空计划或越界读取。

修复：在构建Tree传输计划前统一校验block table行数、cache key可切分性、prefix/input长度及非空key集合；非法输入立即返回模型请求错误，并补充缺失key单测。

## 第七轮：RDMA完成边界

### 18.物理回调与watchdog竞争误报超时

问题：RDMA物理回调先把task标记完成，再竞争RPC完成权；watchdog可能在两步之间抢先返回超时，使已完成传输仍被上层判失败。

修复：物理回调先原子竞争RPC完成权，再无条件通知task物理完成；回调先到时watchdog不再覆盖结果，watchdog先到时晚回调仍能结束task并允许Decode释放目标引用。

### 19.TCP未校验实际payload完整性

问题：TCP只比较声明len与Decode目标大小，未比较实际content长度；截断请求会让设备拷贝越过protobuf缓冲区。重复key、空或无效Decode子block也可能被覆盖或跳过。

修复：要求每个Decode目标key及子block有效，拒绝重复/缺失key，并同时校验声明长度、实际payload长度和目标大小；任一不匹配整体返回BUFFER_MISMATCH，补充截断、重复key及无效目标单测。

## 第八轮：物理后端结果一致性

### 20.任务终态覆盖错误码后仍返回旧错误文案

问题：TCP/RDMA完成回调可能被任务中的取消或超时终态覆盖。原实现会更新响应错误码，却继续使用回调传入的旧文案，导致发送端收到互相矛盾的错误码和原因，影响上层归因。

修复：任务参与完成判定时，同时回读其权威错误码和错误文案；补充取消与超时覆盖成功结果的断言，保证两端收到一致终态。

## 第九轮：终态过期后的迟到回调

### 21.已过请求截止时间的Prefill资源可被重新接纳

问题：终态记录在原请求截止时间清理后，迟到的Prefill回调仍携带已过期截止时间。原实现会把该时间归一化到当前时刻并重新加入资源表，使无消费者的KV再次占用一个hold窗口。

修复：`addResource`入口拒绝携带明确过期截止时间的请求，并触发逐层资源清理；补充终态清理后迟到资源仍不可加入的测试。

## 第十轮：逐层写入结果传播

### 22.逐层调度失败仍返回成功上下文

问题：`P2PConnector::asyncWriteByLayer`忽略worker返回值。拓扑缺失、层映射为空等调度失败时，上层仍得到恒成功的异步上下文，Prefill后续不会提供该层，而发送端只能等到传输截止。

修复：校验worker、上下文和资源；worker拒绝调度时直接返回空上下文，让模型回调链路立即感知失败，并补充调度拒绝测试。

## 第十一轮：Decode广播输入边界

### 23.空READ被判成功且非法目标可触发转换异常

问题：空`READ`会被worker直接当成功；负block id、非法layer/tag则可能进入allocator转换并抛异常。正常调度不会生成这类报文，但远端输入损坏时会出现假成功或RPC线程异常。

修复：入口拒绝空key、空层集合、空块集合、重复key、非法层标签和负block id，再构造本地目标；补充空READ与非法block id测试。

## 第十二轮：全命中控制流

### 24.no_transfer仍广播空READ

问题：全命中时Decode无需拉取KV，但调度端仍向各rank广播不含layer block的READ。READ输入校验收紧后该请求必然失败，使合法no_transfer流程被合并结果判为失败。

修复：no_transfer只发StartLoad，并为本地Decode物理读取分支生成已完成成功结果；普通READ仍要求非空目标集合。更新调度测试，确认全命中不会触发Decode worker广播。

## 第十三轮：修复后静态复核

普通非CP路径未再发现P2及以上问题。重点复核了全命中side-channel收口、部分Tree复用后的目标范围、Decode取消与lease延迟释放、Prefill逐层回调终态、TCP/RDMA完整覆盖及coordinator旁路。CP/CSA适配仍按遗留文档单独处理，不计入本轮可用性结论。

## 第十四轮：Prefill取消边界

### 25.取消广播创建失败后突破传输上界

问题：Prefill等待TP广播时，若客户端已取消或传输截止已到，但取消广播自身创建失败，原循环仍等待HANDLE_READ完成，可能突破单次传输deadline。

修复：取消决定与取消RPC创建结果解耦。取消广播无法创建时生成本地终态结果，立即退出等待并向上层返回取消或超时；取消广播仍保持尽力下发语义。

## 第十五轮：后端覆盖语义一致性

### 26.RDMA错误拒绝Prefill子block超集

问题：约定要求Decode目标全部被覆盖，Prefill多提供的内容可忽略。TCP已按该规则处理，RDMA却要求同一key的子block数量完全相等，合法来源超集会被判为BUFFER_MISMATCH。

修复：RDMA仅在来源子block少于Decode期望时失败，按Decode期望数量构造读操作并忽略尾部额外项；测试改为验证子block超集成功且目标内容正确。

## 第十六轮：首token输出衔接

### 27.Decode重复输出Prefill首token

问题：side-channel首token已由Prefill对外输出。当前Decode把该token写入本地序列前未推进输出游标，后续update会把它再次加入Decode输出队列，客户端可能收到重复token。

修复：恢复原P2P及DecodeRpcServer一致行为，在更新Decode本地序列前调用incLastOutputPos；新增测试校验序列长度与输出游标同步前进且最终对齐。

## 第十七轮：首token完成边界

### 28.单token请求在Decode多执行一步

问题：P2P side-channel应用首token时跳过全部完成判断。`max_new_tokens=1`或首token命中停止条件时，Decode仍从`LOADING_CACHE`回到`WAITING`并被再次调度，可能额外生成token。

修复：恢复首token的正常完成判断；load完成后若已有`GenerateDone`则直接进入`FINISHED`并释放资源，否则回到`WAITING`。删除仅为绕过该判断引入的标志，并补充单token请求测试。

## 第十八轮：后台清理初始化

### 29.Decode清理线程失败后仍启动服务

问题：Decode lease清理线程创建失败时仅记录日志，P2P worker仍返回初始化成功。此后提前取消终态及未被查询的lease记录缺少周期回收，会持续占用内存。

修复：暴露Decode worker的初始化状态，P2P worker创建后立即校验；清理线程不可用时整体初始化失败，避免以缺失生命周期保障的状态接收请求。

## 第十九轮：最终静态复核

普通非CP路径未再发现P2及以上问题。复核范围包括首token完成与输出游标、全命中no_transfer、部分Tree复用、异步read启动及取消竞争、Decode目标延迟释放、Prefill逐层资源收口、TCP/RDMA覆盖语义、deadline与终态回收、初始化失败传播。CP/CSA问题继续按遗留文档处理。

## 第二十轮：首token去重职责复核

### 30.提前推进输出游标导致第二个token丢失

问题：第27项恢复原P2P行为时忽略了当前`DecodeRpcServerNew2`已有重复帧抑制。提前推进游标会让首token不进入Decode队列，RPC层随后把真正的第二个token误认为重复帧并丢弃。

修复：撤销side-channel更新前的游标推进，保留Decode首token重复帧，由RPC层现有逻辑精确丢弃；单测增加输出队列断言。第27项原修复结论由本项更正。

## 第二十一轮：side-channel必需字段

### 31.StartLoad缺少首token仍被判成功

问题：Prefill RPC只检查gRPC状态和业务错误码，成功响应即使没有首token也会进入成功态。Decode随后无法完成Prefill到Decode的序列衔接，可能从错误的context状态继续执行。

修复：解析StartLoad响应后强制校验首token存在；缺失时返回P2P load失败，并补充业务成功但缺少首token的测试。token id为0仍可通过显式存在标志合法传输。

## 第二十二轮：Decode READ结构校验

### 32.重复layer/tag会注册同名接收任务

问题：READ入口只校验单个layer/tag内的key，不拒绝同一请求重复出现相同layer/tag。两项会生成相同的partition传输key，可能覆盖接收任务或让完成状态对应错误的目标集合。

修复：READ解析时维护请求内layer/tag集合，发现重复立即返回调度错误；补充重复layer/tag输入测试。

## 第二十三轮：广播错误码收口

### 33.缺失worker响应被编码为无错误

问题：worker的gRPC调用成功但未携带`p2p_response`时，广播`success()`为false，`errorCode()`却返回`NONE_ERROR`。上层可能结束请求却无法识别业务失败，形成静默截断。

修复：缺少`p2p_response`统一返回worker调用失败并附带rank信息；补充真实广播响应缺字段测试。

## 第二十四轮：逐层事件异常隔离

### 34.CUDA事件查询异常会退出周期线程

问题：逐层发布线程直接调用`torch::Event::query()`，异常未被捕获。无效事件或设备错误可能退出整个周期线程，使其他请求停止发布和清理，并让Prefill长期等待。

修复：在单个context边界捕获标准及未知异常，移除该请求的逐层buffer、记录失败并继续处理后续context；正常事件轮询不变。

## 第二十五轮：Tree关联文档一致性

### 35.全命中P2P失败被误写为可忽略

问题：合并说明称P2P失败只在存在远程block时上报，与当前全命中no-transfer仍需返回首token、释放Prefill资源的强制握手语义矛盾。

修复：文档改为P2P context创建后必须完成；部分命中补KV，全命中执行no-transfer收口，任一失败均上报。同时明确Prefill按Topology分组策略提供数据，Decode只注册Tree未覆盖目标。

## 第二十六轮：非CP端到端静态复核

未再发现P2及以上代码问题。复核了FULL/SWA/LINEAR逻辑槽位映射、Tree部分与全命中、StartLoad双deadline、side-channel终态、Prefill分层发布、Decode READ结构校验、TCP/RDMA完整覆盖与物理完成、异步取消及lease回收、Coordinator旁路和Bazel目标拆分。CP/CSA仍按遗留文档处理。

## 验证说明

- 每轮执行`git diff --check`。
- 已补充资源终态、deadline、no_transfer、覆盖校验、异步分发及逐层错误传播单测。
- 当前未在本机执行Bazel编译或测试；按项目约束，后续应通过指定远端环境和`test-execution`流程验证。
