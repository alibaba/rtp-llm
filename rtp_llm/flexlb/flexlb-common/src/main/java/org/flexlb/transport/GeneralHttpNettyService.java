package org.flexlb.transport;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.HttpContent;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpHeaders;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpObject;
import io.netty.handler.codec.http.HttpResponse;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.LastHttpContent;
import io.netty.handler.timeout.ReadTimeoutException;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.netty.HttpNettyChannelContext;
import org.flexlb.enums.StatusEnum;
import org.flexlb.exception.FlexLBException;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.NettyUtils;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

/**
 * 通用HttpNettyClient请求
 */
@Slf4j
@Component
public class GeneralHttpNettyService {

    private final HttpNettyClientHandler nettyClient;
    private final Scheduler httpRequestScheduler;

    public static final ThreadPoolExecutor httpRequestExecutor = new ThreadPoolExecutor(10 * Runtime.getRuntime()
            .availableProcessors(), 15 * Runtime.getRuntime()
            .availableProcessors(), 60L, TimeUnit.SECONDS, new SynchronousQueue<>(), new NamedThreadFactory("req-thread"),
            // 拒绝策略：队列满时由提交任务的线程执行（避免任务丢失）
            new ThreadPoolExecutor.CallerRunsPolicy());

    public GeneralHttpNettyService(HttpNettyClientHandler nettyClient) {
        this.nettyClient = nettyClient;
        httpRequestScheduler = Schedulers.fromExecutor(httpRequestExecutor);
    }

    public <Request, Result> Mono<Result> request(Request request, URI uri, String path, Class<Result> responseClz) {
        return this.doRequest(request, uri, path, null, responseClz);
    }

    public <Request, Result> Mono<Result> request(Request request, URI uri, String path, HttpHeaders headers, Class<Result> responseClz) {

        return Mono.fromFuture(this.doRequest(request, uri, path, headers, responseClz).toFuture());
    }

    public <Request, Result> Mono<Result> doRequest(Request request, URI uri, String path, HttpHeaders headers, Class<Result> responseClz) {
        return Mono.just(request)
                .map(ctx -> HttpNettyChannelContext.<Result>builder()
                        .request(request)
                        .readCallback((resultHttpNettyChannelContext, httpObject)
                                -> handleNettyMessage(resultHttpNettyChannelContext, httpObject, responseClz))
                        .errorCallback(this::handlerNettyError)
                        .byteDataList(new ArrayList<>(1 << 4))
                        .byteDataSize(new LongAdder())
                        .build())
                .flatMap(nettyCtx -> connectBackend(nettyCtx, uri, path).publishOn(httpRequestScheduler)
                        .flatMap(nettyContext -> executeHttpRequest(nettyContext, uri, path, headers
                        )));
    }

    private <Result> Mono<HttpNettyChannelContext<Result>> connectBackend(HttpNettyChannelContext<Result> nettyCtx,
                                                                          URI uri, String path) {
        // 发起连接
        int defaultPort = "http".equalsIgnoreCase(uri.getScheme()) ? 80 : 443;
        int port = uri.getPort() == -1 ? defaultPort : uri.getPort();
        ChannelFuture channelFuture = nettyClient.connect(uri.getHost(), port);

        // 绑定上下文
        Channel channel = channelFuture.channel();
        nettyCtx.setChannel(channel);
        nettyClient.setNettyChannelContext(channel, nettyCtx);

        // 处理未来并返回 Mono.fromFuture
        CompletableFuture<HttpNettyChannelContext<Result>> future = new CompletableFuture<>();
        channelFuture.addListener((ChannelFutureListener) cf -> {
            try {
                if (cf.isSuccess()) {
                    future.complete(nettyCtx);
                } else {
                    future.completeExceptionally(StatusEnum.INTERNAL_ERROR.toException("failed to connect, uri=" + uri + ", path=" + path, cf.cause()));
                }
            } catch (Exception e) {
                future.completeExceptionally(StatusEnum.INTERNAL_ERROR.toException("failed to connect, uri=" + uri + ", path=" + path, e));
            }
        });
        return Mono.fromFuture(future);
    }

    private <Result> Mono<Result> executeHttpRequest(HttpNettyChannelContext<Result> nettyCtx, URI uri, String path, HttpHeaders headers) {
        return Flux.<Result>create(sink -> {
            nettyCtx.setSink(sink);
            DefaultFullHttpRequest request = buildRequest(nettyCtx, uri, path, headers);
            nettyCtx.getChannel().writeAndFlush(request);
        }).last();
    }

    private <Result> DefaultFullHttpRequest buildRequest(HttpNettyChannelContext<Result> nettyCtx, URI uri, String path, HttpHeaders headers) {
        DefaultFullHttpRequest request = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, path);

        String body = JsonUtils.toStringOrEmpty(nettyCtx.getRequest());
        request.content().writeBytes(body.getBytes(StandardCharsets.UTF_8));
        if (headers == null) {

            request.headers().set(HttpHeaderNames.HOST, Objects.requireNonNull(uri).getHost());
            request.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.KEEP_ALIVE);
            request.headers().set(HttpHeaderNames.CONTENT_TYPE, "application/json");
            request.headers().set(HttpHeaderNames.CONTENT_LENGTH, request.content().readableBytes());
        } else {
            request.headers().setAll(headers);
        }
        return request;
    }

    private <Result> void handlerNettyError(HttpNettyChannelContext<Result> nettyCtx, Throwable e) {
        if (e instanceof ReadTimeoutException) {
            nettyCtx.getSink().error(StatusEnum.READ_TIME_OUT.toException());
        } else {
            nettyCtx.getSink()
                    .error(StatusEnum.NETTY_CATCH_ERROR.toException("sync load balance unexpected netty " + "error " + "happened, exception: ",
                            e));
        }
        nettyCtx.getChannel().close();
    }

    private <Result> void handleNettyMessage(HttpNettyChannelContext<Result> nettyCtx, HttpObject obj,
                                             Class<Result> responseClz) {
        if (nettyCtx.getSink().isCancelled()) {
            NettyUtils.finish(nettyCtx);
            log.error("sink canceled, finish netty");
            return;
        }
        if (nettyCtx.isFinish()) {
            return;
        }

        if (obj instanceof HttpResponse response) {
            nettyCtx.setHttpResp(response);
        } else if (obj instanceof HttpContent) {
            handleNettyChunk(nettyCtx, obj, responseClz);
        } else {
            log.error("uncatchable message types {}", obj);
        }
    }

    private <Result> void handleNettyChunk(HttpNettyChannelContext<Result> nettyCtx, HttpObject obj,
                                           Class<Result> responseClz) {

        // 校验当前的http的response是否是200，如果是非200，代表是异常情况，直接解析chunk返回错误即可。
        int httpStatusCode = NettyUtils.getHttpStatusCode(nettyCtx);

        // 如果状态码不是200，代表是异常情况，直接解析chunk返回错误即可。
        if (httpStatusCode != StatusEnum.SUCCESS.getCode()) {
            NettyUtils.cacheBuffer(nettyCtx, obj);
            String body = NettyUtils.readBody(nettyCtx);
            NettyUtils.finishNettyWithException(nettyCtx,
                    new RuntimeException("http error, httpStatusCode=" + httpStatusCode + ", body=" + body));
            return;
        }

        // 添加到缓冲区
        NettyUtils.cacheBuffer(nettyCtx, obj);

        // 非流式调用, 直到最后一个 chunk 到达, 才开始解析
        if (!(obj instanceof LastHttpContent)) {
            return;
        }

        try {
            byte[] bodyBytes = NettyUtils.getBodyBytes(nettyCtx);
            nettyCtx.getByteDataList().clear();
            nettyCtx.getByteDataSize().reset();

            // 非流式调用，如果数据为空，那么忽略
            // 正常情况不会出现bodyBytes为空的情况
            // 当前仅在引擎异常断联的时候会出现, 因为HttpObjectAggregator聚合的原因, 会返回一个空body的FullResponse
            if (bodyBytes.length == 0) {
                return;
            }

            Result response;
            try {
                response = JsonUtils.toObject(bodyBytes, responseClz);
            } catch (Throwable e) {
                throw StatusEnum.INTERNAL_ERROR.toException(e);
            }
            nettyCtx.getSink().next(response);
            // 上游流结束
            nettyCtx.getSink().complete();
            // 下游netty结束
            NettyUtils.finish(nettyCtx);

        } catch (FlexLBException e) {
            nettyCtx.getSink().error(e);
            NettyUtils.finish(nettyCtx);
        }
    }
}
