package org.flexlb.sync.runner;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelPipeline;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpContent;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpObject;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpResponse;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.LastHttpContent;
import io.netty.handler.timeout.ReadTimeoutException;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.consistency.WorkerStatusRequest;
import org.flexlb.constant.CommonConstants;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.netty.HttpNettyChannelContext;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.StatusEnum;
import org.flexlb.exception.WhaleException;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.StatusQueryDisposableCleaner;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.transport.HttpNettyClientHandler;
import org.flexlb.util.IdUtils;
import org.flexlb.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.FluxSink;
import reactor.core.publisher.Mono;

import static org.flexlb.sync.StatusQueryDisposableCleaner.CLEAN_DISPOSABLE_PERIOD;

public class HttpStatusCheckRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String ipPort;

    private final String modelName;

    private final String site;

    private final ConcurrentHashMap<String/*ipPort*/, WorkerStatus> workerStatuses;

    private final EngineHealthReporter engineHealthReporter;

    private final HttpNettyClientHandler nettyClient;
    private final String ip;
    private final int port;
    private final long startTime = System.currentTimeMillis();
    private final String id = IdUtils.fastUuid();
    private final String group;
    private final CacheAwareService localKvCacheAwareManager;

    public HttpStatusCheckRunner(String modelName, String ipPort, String site, String group,
                             ConcurrentHashMap<String/*ip*/, WorkerStatus> workerStatuses,
                             EngineHealthReporter engineHealthReporter,
                             HttpNettyClientHandler nettyClient,
                             CacheAwareService localKvCacheAwareManager) {
        this.ipPort = ipPort;
        String[] split = ipPort.split(":");
        this.ip = split[0];
        this.port = Integer.parseInt(split[1]);
        this.modelName = modelName;
        this.workerStatuses = workerStatuses;
        this.site = site;
        this.engineHealthReporter = engineHealthReporter;
        this.nettyClient = nettyClient;
        this.group = group;
        this.localKvCacheAwareManager = localKvCacheAwareManager;
    }

    @Override
    public void run() {
        logger.info("HttpStatusCheckRunner run");
        // 如果异步状态检查DisposableMap大小超过30000，则清理
        Map<String, Disposable> queryStatusDisposableMap = EngineWorkerStatus.queryStatusDisposableMap;
        logger.info("queryStatusDisposableMap size: {}", queryStatusDisposableMap.size());
        if (queryStatusDisposableMap.size() >= 30000) {
            long lastUpdateTime = StatusQueryDisposableCleaner.updateTime.get();
            long currentTimeMillis = System.currentTimeMillis();
            if (currentTimeMillis - lastUpdateTime >= CLEAN_DISPOSABLE_PERIOD) {
                StatusQueryDisposableCleaner.updateTime.set(currentTimeMillis);
                log("clean queryStatusDisposableMap, size: " + queryStatusDisposableMap.size());
                StatusQueryDisposableCleaner.cleanDisposable();
            }
        }
        long startTime = System.currentTimeMillis();
        // 添加参数校验
        if (StringUtils.isBlank(ipPort)) {
            logger.info("Invalid ipPort: {}",ipPort);
        }else{
            logger.info(" ipPort: {}" , ipPort);
        }
        WorkerStatus workerStatus = workerStatuses.get(ipPort);
        long cacheVersion = Optional.ofNullable(workerStatus)
                .map(WorkerStatus::getCacheStatus)
                .map(CacheStatus::getVersion)
                .orElse(-1L);
        long latestFinishedTaskVersion = Optional.ofNullable(workerStatus)
                .map(WorkerStatus::getLatestFinishedTaskVersion)
                .map(AtomicLong::get)
                .orElse(-1L);
        WorkerStatusRequest workerStatusRequest = new WorkerStatusRequest(cacheVersion, latestFinishedTaskVersion);
        Disposable disposable = launchStatusCheck(ipPort, workerStatusRequest).subscribe(
                updateContent -> handleStatusResponse(updateContent, startTime),
                ex -> log("status check failed:ipPort:" + ipPort + ",  with exception: "  + ex.getMessage()));
        queryStatusDisposableMap.put(System.currentTimeMillis() + "@" + id, disposable);
    }

    private void handleStatusResponse(WorkerStatusResponse updateContent, long startTime) {
        try {
            logger.info("handleStatusResponse start ");
            if (updateContent == null) {
                logger.info("query engine worker status, response body is null");
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.RESPONSE_NULL);
                return;
            }
            engineHealthReporter.reportStatusCheckRemoteInfo(modelName, ipPort, updateContent.getRole(), startTime);
            if (updateContent.getMessage() != null) {
                logger.info("query engine worker status, msg={}", updateContent.getMessage());
                return;
            }
            Long responseVersion = updateContent.getStatusVersion();
            if(responseVersion == 0L){
                logger.info("workerStatuses.get(ip) is null");
                return;
            }
            WorkerStatus workerStatus = workerStatuses.get(ipPort);
            if (workerStatus == null) {
                logger.info("workerStatuses.get(ip) is null, now ip is {}", ip);
                workerStatus = new WorkerStatus();
                workerStatus.setIp(ip);
                workerStatuses.put(ipPort, workerStatus);
            }
            workerStatus.setPort(port);
            workerStatus.setSite(site);
            workerStatus.setRole(updateContent.getRole());
            workerStatus.setGroup(group);
            Long currentVersion = workerStatus.getStatusVersion();
            // logger.info("currentVersion: {}, responseVersion: {} ipPort : {}", currentVersion, responseVersion, ipPort);
            if (currentVersion >= responseVersion) {
                logger.info("query engine worker status, version is not updated, currentVersion: {}, responseVersion: {}",
                    currentVersion, responseVersion);
                return;
            }

            workerStatus.setAvailableConcurrency(updateContent.getAvailableConcurrency());
            workerStatus.setStepLatencyMs(updateContent.getStepLatencyMs());
            workerStatus.setIterateCount(updateContent.getIterateCount());
            workerStatus.setDpSize(updateContent.getDpSize());
            workerStatus.setTpSize(updateContent.getTpSize());
            workerStatus.setAlive(updateContent.isAlive());
            workerStatus.setVersion(String.valueOf(updateContent.getVersion()));
            workerStatus.setStatusVersion(responseVersion);
            workerStatus.setRunningTaskList(updateContent.getRunningTaskInfo());
            workerStatus.clearFinishedTaskAndTimeoutTask(updateContent.getFinishedTaskList());

            ConcurrentHashMap<Long, TaskInfo> localTaskMap = workerStatus.getLocalTaskMap();
            CacheStatus cacheStatus = updateContent.getCacheStatus();
            long cache_free = cacheStatus.getAvailableKvCache();
            long cache_use = cacheStatus.getTotalKvCache() - cache_free;


            long local_cache_free = workerStatus.getKvCacheFree().intValue();
            logger.info("remote_cache_free : {}, local_cache_free: {} ", cache_free, local_cache_free);
            int local_cache_size = localTaskMap.size();
            logger.info("local_cache_size : {}", local_cache_size);
            if(local_cache_size == 0){
                workerStatus.getKvCacheUsed().getAndSet(cache_use);
                workerStatus.getKvCacheFree().getAndSet(cache_free);
                workerStatus.getRunningQueueTime().getAndSet(0);
            } else {
                long tmp_used = 0;
                for(Map.Entry<Long, TaskInfo> entry : localTaskMap.entrySet()) {
                    tmp_used += entry.getValue().estimatePrefillTime();
                }
                cache_use += tmp_used;
                cache_free -= tmp_used;
                workerStatus.getKvCacheUsed().getAndSet(cache_use);
                workerStatus.getKvCacheFree().getAndSet(cache_free);
                if (RoleType.PREFILL.matches(updateContent.getRole())) {
                    if (workerStatus.getRunningQueueTime().get() > tmp_used) {
                        workerStatus.getRunningQueueTime().getAndSet(tmp_used);
                    }
                }else{
                    logger.info("running_queue_time : {} , tmp_used : {}" , workerStatus.getRunningQueueTime().get(), tmp_used);
                }
            }

            if (cacheStatus.getVersion() != Optional.ofNullable(workerStatus.getCacheStatus()).map(CacheStatus::getVersion).orElse(-1L)) {
                workerStatus.setCacheStatus(cacheStatus);
            }


            // 设置当前时间的3s后为过期时间
            workerStatus.getExpirationTime().set(System.currentTimeMillis() + 3000);

            logger.info("IP:{} Port:{}, running_queue_tokens:{}, cache_used:{}, cache_free:{}, cost:{}",
                    workerStatus.getIp(),
                    workerStatus.getPort(),
                    workerStatus.getRunningQueueTime(),
                    workerStatus.getKvCacheUsed(),
                    workerStatus.getKvCacheFree(),
                    System.currentTimeMillis() - startTime);

            engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus);
            workerStatus.getLastUpdateTime().set(System.currentTimeMillis());
            
            // 更新KV Cache本地缓存
            try {
                WorkerCacheUpdateResult workerCacheUpdateResult = localKvCacheAwareManager.updateEngineBlockCache(workerStatus);
                if (!workerCacheUpdateResult.isSuccess()) {
                    logger.warn("Failed to update worker cache for IP: {}, error: {}", workerStatus.getIp(),
                        workerCacheUpdateResult.getErrorMessage());
                }
                logger.debug("Successfully updated worker cache for IP: {}", workerStatus.getIp());
            } catch (Exception e) {
                logger.warn("Failed to update worker cache for IP: {}, error: {}", workerStatus.getIp(), e.getMessage());
            }

        } catch (Throwable e) {
            log("engine worker status check exception, msg: " + e.getMessage(), e);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR);
        }
    }

    public Mono<WorkerStatusResponse> launchStatusCheck(String ipPort, WorkerStatusRequest workerStatusRequest) {
        URI uri = getUri(ipPort);
        if (uri == null) {
            return null;
        }
        return Mono.just(HttpNettyChannelContext.<WorkerStatusResponse>builder()
                        .requestCtx(null)
                        .readCallback(this::handleNettyMessage)
                        .errorCallback(this::handlerNettyError)
                        .channelInactiveCallback(this::handleChannelInactive)
                        .channelEnhanceCallback(this::channelEnhance)
                        .byteDataSize(new LongAdder())
                        .buffer(new ArrayList<>(1 << 9))
                        .byteDataList(new ArrayList<>(1 << 4))
                        .build())
                .flatMap(nettyCtx -> connectBackend(nettyCtx, uri)
                        .flatMap(nettyContext -> executeHttpRequest(nettyContext, uri, workerStatusRequest)));
    }

    private Mono<WorkerStatusResponse> executeHttpRequest(HttpNettyChannelContext<WorkerStatusResponse> nettyCtx,
                                                         URI uri, WorkerStatusRequest workerStatusRequest) {
        return Flux.<WorkerStatusResponse>create(sink -> {
            nettyCtx.setSink(sink);
            DefaultFullHttpRequest request = buildRequest(uri, workerStatusRequest);
            nettyCtx.getChannel().writeAndFlush(request);
        }).last().doOnCancel(() -> log("cancel query engine worker status"));
    }

    private void handleChannelInactive(HttpNettyChannelContext<WorkerStatusResponse> nettyCtx) {
        finish(nettyCtx);
        Optional.of(nettyCtx).map(HttpNettyChannelContext::getSink).ifPresent(FluxSink::complete);
    }

    private void channelEnhance(HttpNettyChannelContext<WorkerStatusResponse> nettyCtx) {
        ChannelPipeline pipeline = nettyCtx.getChannel().pipeline();
        // 零拷贝优化:聚合缓冲数据, 避免多次复制分片数据
        // 使用聚合, 最大聚合10MB , 减少数据在内核态和用户态之间切换
        int bigResponseSizeThreshold = 10485760;
        pipeline.addAfter(CommonConstants.CODEC, CommonConstants.AGGREGATOR,
                new HttpObjectAggregator(bigResponseSizeThreshold));// 10MB
    }

    private DefaultFullHttpRequest buildRequest(URI uri, WorkerStatusRequest workerStatusRequest) {
        String jsonBody;
        try {
            jsonBody = new ObjectMapper().writeValueAsString(workerStatusRequest);
        } catch (JsonProcessingException e) {
            log("when build request: to json failed, workerStatusRequest:"+  workerStatusRequest.toString(), e);
            jsonBody = "{}";
        }
        log("sed request" + uri.getHost() + ",jsonBody:"+  jsonBody);
        ByteBuf content = Unpooled.copiedBuffer(jsonBody, StandardCharsets.UTF_8);
        DefaultFullHttpRequest request = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST,
                "/worker_status", content);
        request.headers().set(HttpHeaderNames.HOST, Objects.requireNonNull(uri).getHost());
        request.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.KEEP_ALIVE);
        request.headers().set(HttpHeaderNames.CONTENT_TYPE, "application/json");

        return request;
    }

    private Mono<HttpNettyChannelContext<WorkerStatusResponse>> connectBackend(HttpNettyChannelContext<WorkerStatusResponse> nettyCtx, URI uri) {
        // 发起连接
        int port = uri.getPort() == -1 ? 80 : uri.getPort();
        ChannelFuture channelFuture = nettyClient.connect(uri.getHost(), port);

        // 绑定上下文
        Channel channel = channelFuture.channel();
        nettyCtx.setChannel(channel);
        nettyClient.setNettyChannelContext(channel, nettyCtx);

        // 处理未来并返回 Mono.fromFuture
        CompletableFuture<HttpNettyChannelContext<WorkerStatusResponse>> future = new CompletableFuture<>();
        channelFuture.addListener((ChannelFutureListener) cf -> {
            try {
                if (cf.isSuccess()) {
                    future.complete(nettyCtx);
                } else {
                    future.completeExceptionally(new RuntimeException("failed to connect engine ip"));
                }
            } catch (Exception e) {
                log("failed to connect engine ip, e", e);
                future.completeExceptionally(
                        StatusEnum.INTERNAL_ERROR.toException("failed to connect engine ip, msg=" + e.getMessage()));
            }
        });
        return Mono.fromFuture(future);
    }

    private void handlerNettyError(HttpNettyChannelContext<WorkerStatusResponse> nettyCtx, Throwable e) {
        if (e instanceof ReadTimeoutException) {
            Optional.ofNullable(nettyCtx.getSink()).ifPresent(s -> s.error(StatusEnum.READ_TIME_OUT.toException()));
        } else {
            log("engine status check unexpected netty error happened", e);
            Optional.ofNullable(nettyCtx.getSink()).ifPresent(s -> s.error(
                    StatusEnum.NETTY_CATCH_ERROR.toException("engine status check unexpected netty " + "error " +
                            "happened, exception: ", e)));
        }
        nettyCtx.getChannel().close();
    }

    private void handleNettyMessage(HttpNettyChannelContext<WorkerStatusResponse> nettyCtx, HttpObject obj) {
        if (nettyCtx.getSink().isCancelled()) {
            finish(nettyCtx);
            cacheBuffer(nettyCtx, obj);
            String body = readBody(nettyCtx);
            log("sink canceled, finish netty" + ", body: " + body);
            return;
        }
        if (nettyCtx.isFinish()) {
            return;
        }

        // FullHttpResponse和pipeline-aggregator配合使用, FullHttpResponse包含请求头和响应体全部
        if (obj instanceof FullHttpResponse) {
            FullHttpResponse fullHttpResponse = (FullHttpResponse) obj;
            nettyCtx.setHttpResp(fullHttpResponse);
            handleNettyChunk(nettyCtx, fullHttpResponse);
        }
        // 如果不是 FullHttpResponse, 那么请求头和请求体(分片)是多次来的
        else if (obj instanceof HttpResponse) {
            nettyCtx.setHttpResp((HttpResponse) obj);
        } else if (obj instanceof HttpContent) {
            handleNettyChunk(nettyCtx, obj);
        } else {
            log("uncatchable message types " + obj);
        }
    }

    private void handleNettyChunk(HttpNettyChannelContext<WorkerStatusResponse> nettyCtx, HttpObject obj) {

        cacheBuffer(nettyCtx, obj);
        // 可能出现拆包问题，等到最后一个包统一处理
        if (!(obj instanceof LastHttpContent)) {
            return;
        }
        String body = readBody(nettyCtx);

        try {
            WorkerStatusResponse response;
            // 校验当前的http的response是否是200，如果是非200，代表是异常情况，直接解析chunk返回错误即可。
            //noinspection deprecation
            int httpStatusCode = Optional.of(nettyCtx)
                    .map(HttpNettyChannelContext::getHttpResp)
                    .map(HttpResponse::getStatus)
                    .map(HttpResponseStatus::code)
                    .orElse(-99);

            if (httpStatusCode != StatusEnum.SUCCESS.getCode()) {
                response = new WorkerStatusResponse();
                response.setMessage(body);
            } else {
                try {
                    response = JsonUtils.toObject(body, new TypeReference<WorkerStatusResponse>() {
                    });
                } catch (Throwable e) {
                    throw StatusEnum.INTERNAL_ERROR.toException(e);
                }
            }

            nettyCtx.getSink().next(response);

            // 上游流结束
            nettyCtx.getSink().complete();

            // 下游netty结束
            finish(nettyCtx);

        } catch (WhaleException e) {
            log("query engine worker status handle response error! ", e);
            nettyCtx.getSink().error(e);
            finish(nettyCtx);
            log("query engine worker status finish netty with exception, ", e);
        }
    }

    public String readBody(HttpNettyChannelContext<WorkerStatusResponse> nettyCtx) {
        byte[] mergedData = getBodyBytes(nettyCtx);
        return new String(mergedData, StandardCharsets.UTF_8);
    }

    public byte[] getBodyBytes(HttpNettyChannelContext<WorkerStatusResponse> nettyCtx) {
        List<HttpNettyChannelContext.ByteData> byteDataList = nettyCtx.getByteDataList();
        // 如果只有一个 chunk, 那么不需要合并, 直接返回
        if (byteDataList.size() == 1) {
            return byteDataList.get(0).getData();
        }
        long totalBufferSize = nettyCtx.getByteDataSize().sum();
        byte[] mergedData = new byte[(int) totalBufferSize];
        int index = 0;
        for (HttpNettyChannelContext.ByteData byteData : byteDataList) {
            byte[] data = byteData.getData();
            System.arraycopy(data, 0, mergedData, index, data.length);
            index += data.length;
        }
        return mergedData;
    }

    private void cacheBuffer(HttpNettyChannelContext<WorkerStatusResponse> nettyCtx, HttpObject obj) {

        ByteBuf content = ((HttpContent) obj).content();
        nettyCtx.getByteDataSize().add(content.readableBytes());
        byte[] buffer = new byte[content.readableBytes()];
        content.getBytes(content.readerIndex(), buffer);

        List<HttpNettyChannelContext.ByteData> byteDataList = nettyCtx.getByteDataList();
        byteDataList.add(new HttpNettyChannelContext.ByteData(buffer));
    }

    private URI getUri(String ipPort) {
        try {
            String url = "http://" + ipPort;
            return new URI(url);
        } catch (Exception e) {
            log("get uri failed", e);
            return null;
        }
    }

    private void finish(HttpNettyChannelContext<WorkerStatusResponse> nettyCtx) {
        nettyCtx.setFinish(true);
        nettyCtx.getChannel().close();
    }

    private void log(String msg) {
        logger.info("[{}][{}][{}][{}][{}ms]: {}",
                id,
                site,
                ipPort,
                modelName,
                System.currentTimeMillis() - startTime,
                msg);
    }

    private void log(String msg, Throwable e) {
        logger.info("[{}][{}][{}][{}][{}ms]: {}",
                id,
                site,
                ipPort,
                modelName,
                System.currentTimeMillis() - startTime,
                msg,
                e);
    }
}
