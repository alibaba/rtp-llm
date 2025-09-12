package org.flexlb.constant;

/**
 * Metric constants for monitoring and observability
 * Use with standard monitoring libraries like Micrometer/Prometheus
 */
public class MetricConstant {

    /* ------------------------ 引擎状态统计 -------------------------- */

    /**
     * 引擎状态检查更新周期
     */
    public static final String ENGINE_STATUS_CHECK_SUCCESS_PERIOD = "app.engine.health.check.success.period";

    /**
     * 引擎worker数量
     */
    public static final String ENGINE_WORKER_NUMBER = "app.engine.health.check.engine.worker.number";

    public static final String ENGINE_PREFILL_WORKER_NUMBER = "app.engine.health.check.engine.prefill.worker.number";

    public static final String ENGINE_DECODE_WORKER_NUMBER = "app.engine.health.check.engine.decode.worker.number";

    /**
     * VipServer Client 请求结果
     */
    public static final String ENGINE_NUMBER_VIP_RESULT = "app.engine.health.check.engine.worker.number.vipserver.result";

    /**
     * 引擎worker剩余可用并发数
     */
    public static final String ENGINE_STATUS_AVAILABLE_CONCURRENCY = "app.engine.health.check.available.concurrency";

    public static final String ENGINE_STATUS_VISITOR_RT = "app.engine.health.check.visitor.rt";

    public static final String ENGINE_STATUS_VISITOR_SUCCESS_QPS = "app.engine.health.check.visitor.qps";

    /**
     * 引擎状态检查失败信息
     */
    public static final String ENGINE_STATUS_CHECK_FAIL = "app.engine.health.check.fail";

    /**
     * Master负载均衡服务的总qps
     */
    public static final String ENGINE_BALANCING_MASTER_ALL_QPS = "app.engine.balancing.master.all.qps";

    public static final String ENGINE_BALANCING_MASTER_SCHEDULE_RT = "app.engine.balancing.master.all.rt";

    /**
     * Master负载均衡服务的失败qps
     */
    public static final String ENGINE_BALANCING_MASTER_FAIL_QPS = "app.engine.balancing.master.fail.qps";

    /**
     * prefill 负载均衡 筛选节点QPS
     */
    public static final String PREFILL_BALANCE_SELECT_QPS = "app.engine.prefill.balance.select.qps";

    /**
     * prefill 负载均衡 筛选节点失败QPS
     */
    public static final String PREFILL_BALANCE_SELECT_FAIL_QPS = "app.engine.prefill.balance.select.qps.error";

    /**
     * prefill 负载均衡 筛选节点花费的时间
     */
    public static final String PREFILL_BALANCE_TOKENIZE_COST = "app.engine.prefill.balance.select.cost";

    /**
     * 引擎队列等待时间
     */
    public static final String ENGINE_RUNNING_QUEUE_TIME = "app.engine.health.check.running.queue.time";

    /**
     * prefill master节点监控
     */
    public static final String PREFILL_MASTER_NODE = "app.engine.prefill.balance.master.node";

    /**
     * prefill master节点事件监控
     */
    public static final String PREFILL_MASTER_EVENT = "app.engine.prefill.balance.master.event";

    /**
     * 负载均衡服务的线程池状态
     */
    public static final String ENGINE_BALANCING_THREAD_POOL_INFO = "app.engine.balancing.thread.pool.info";

    /**
     * 负载均衡服务的 NioEvenLoopGroup 状态
     */
    public static final String ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO = "app.engine.balancing.event.loop.group.info";

    /**
     * 获取引擎worker信息服务的step latency 方差
     */
    public final static String ENGINE_WORKER_INFO_STEP_LATENCY_VAR = "app.engine.worker.info.step.latency.var";


    public final static String ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR = "app.engine.worker.info.running.query.len.var";

    /* ------------------------ 缓存健康监控 -------------------------- */

    /**
     * 每个引擎局部缓存的数量
     */
    public static final String CACHE_ENGINE_LOCAL_COUNT = "app.cache.engine.local.count";

    /**
     * 全局缓存的总数量
     */
    public static final String CACHE_GLOBAL_TOTAL_COUNT = "app.cache.global.total.count";

    /**
     * 局部缓存占用的字节数
     */
    public static final String CACHE_ENGINE_LOCAL_BYTES = "app.cache.engine.local.bytes";

    /**
     * 全局缓存占用的字节数
     */
    public static final String CACHE_GLOBAL_BYTES = "app.cache.global.bytes";

    /**
     * 缓存命中数量
     */
    public static final String CACHE_HIT_COUNT = "app.cache.hit.count";

    /**
     * 缓存命中百分比
     */
    public static final String CACHE_HIT_RATIO = "app.cache.hit.ratio";

    /**
     * 缓存请求总数
     */
    public static final String CACHE_REQUEST_TOTAL = "app.cache.request.total";
    
    /**
     * 查找匹配引擎的响应时间
     */
    public static final String CACHE_FIND_MATCHING_ENGINES_RT = "app.cache.find.matching.engines.rt";
    
    /**
     * 更新缓存的响应时间
     */
    public static final String CACHE_UPDATE_ENGINE_BLOCK_CACHE_RT = "app.cache.update.engine.block.cache.rt";
    
    /**
     * 缓存状态检查响应时间
     */
    public static final String CACHE_STATUS_CHECK_VISITOR_RT = "app.cache.status.check.visitor.rt";

    public static final String CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS = "app.cache.status.check.visitor.success.qps";
    
    /**
     * 缓存状态检查成功周期
     */
    public static final String CACHE_STATUS_CHECK_SUCCESS_PERIOD = "app.cache.status.check.success.period";
    
    /**
     * 缓存状态检查失败信息
     */
    public static final String CACHE_STATUS_CHECK_FAIL = "app.cache.status.check.fail";
    
    /**
     * 缓存块大小
     */
    public static final String CACHE_BLOCK_SIZE = "app.cache.block.size";
    
    /**
     * 缓存diff计算中新增块的数量
     */
    public static final String CACHE_DIFF_ADDED_BLOCKS_SIZE = "app.cache.diff.added.blocks.size";
    
    /**
     * 缓存diff计算中移除块的数量
     */
    public static final String CACHE_DIFF_REMOVED_BLOCKS_SIZE = "app.cache.diff.removed.blocks.size";
    
    /**
     * 引擎视图Map的大小（即当前有多少个引擎）
     */
    public static final String CACHE_ENGINE_VIEWS_MAP_SIZE = "app.cache.engine.views.map.size";
    
    /* ------------------------ gRPC连接池监控 -------------------------- */
    
    /**
     * gRPC连接池中的连接数量
     */
    public static final String GRPC_CHANNEL_POOL_SIZE = "app.grpc.channel.pool.size";
}
