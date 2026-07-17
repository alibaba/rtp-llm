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
import org.flexlb.exception.HttpErrorResponseException;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.NettyUtils;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.FluxSink;
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
import java.util.function.Supplier;

/**
 * Generic HttpNettyClient request service
 */
@Slf4j
@Component
public class GeneralHttpNettyService {

    private final HttpNettyClientHandler nettyClient;
    private final Scheduler httpRequestScheduler;

    public static final ThreadPoolExecutor httpRequestExecutor = new ThreadPoolExecutor(10 * Runtime.getRuntime()
            .availableProcessors(), 15 * Runtime.getRuntime()
            .availableProcessors(), 60L, TimeUnit.SECONDS, new SynchronousQueue<>(), new NamedThreadFactory("req-thread"),
            // Rejection policy: execute by submitting thread when queue is full (avoid task loss)
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
                        .channelInactiveCallback(this::handleChannelInactive)
                        .byteDataList(new ArrayList<>(1 << 4))
                        .byteDataSize(new LongAdder())
                        .build())
                .flatMap(nettyCtx -> connectBackend(nettyCtx, uri, path).publishOn(httpRequestScheduler)
                        .flatMap(nettyContext -> executeHttpRequest(nettyContext, uri, path, headers
                        )));
    }

    private <Result> Mono<HttpNettyChannelContext<Result>> connectBackend(HttpNettyChannelContext<Result> nettyCtx,
                                                                          URI uri, String path) {
        // Initiate connection
        int defaultPort = "http".equalsIgnoreCase(uri.getScheme()) ? 80 : 443;
        int port = uri.getPort() == -1 ? defaultPort : uri.getPort();
        ChannelFuture channelFuture = nettyClient.connect(uri.getHost(), port);

        // Bind context
        Channel channel = channelFuture.channel();
        nettyCtx.setChannel(channel);
        nettyClient.setNettyChannelContext(channel, nettyCtx);

        // Handle future and return Mono.fromFuture
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
            if (nettyCtx.installSink(sink)) {
                // The exchange ended before the sink existed, so whoever ended it had nothing to
                // terminate and left the duty here.
                sink.error(StatusEnum.ENGINE_ABNORMAL_DISCONNECT_EXCEPTION
                        .toException("channel closed before request could be written"));
                return;
            }
            DefaultFullHttpRequest request = buildRequest(nettyCtx, uri, path, headers);
            // A write onto a channel that died between the connect and here fails its promise and
            // fires nothing else: the peer is already gone, so no inbound frame and no read timeout
            // will ever terminate the sink.
            nettyCtx.getChannel().writeAndFlush(request).addListener((ChannelFutureListener) future -> {
                if (!future.isSuccess()) {
                    failSink(nettyCtx, () -> StatusEnum.ENGINE_ABNORMAL_DISCONNECT_EXCEPTION
                            .toException("failed to write request, uri=" + uri + ", path=" + path, future.cause()));
                }
            });
        }).last();
    }

    /**
     * The peer disconnected mid-exchange. Both aggregation paths (200 and non-200) terminate the
     * sink only on {@link LastHttpContent}, which a disconnected peer will never send, and the
     * read-timeout handler stops firing once the channel is inactive — so without failing the sink
     * here the caller would wait forever.
     */
    private <Result> void handleChannelInactive(HttpNettyChannelContext<Result> nettyCtx) {
        failSink(nettyCtx, () -> StatusEnum.ENGINE_ABNORMAL_DISCONNECT_EXCEPTION
                .toException("channel closed before the response completed"));
    }

    /**
     * Fails the sink if this caller wins the exclusive termination claim, so the several paths that
     * can end an exchange (disconnect, netty error, failed write) cannot double-terminate it or,
     * worse, each assume another one did. The error is built only by the winner — the losing path
     * is the common one (a completed exchange closes its own channel) and must stay allocation-free.
     */
    private <Result> void failSink(HttpNettyChannelContext<Result> nettyCtx, Supplier<Throwable> error) {
        FluxSink<Result> sink = nettyCtx.claimTermination();
        if (sink == null || sink.isCancelled()) {
            return;
        }
        sink.error(error.get());
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
            failSink(nettyCtx, StatusEnum.READ_TIME_OUT::toException);
        } else {
            failSink(nettyCtx, () -> StatusEnum.NETTY_CATCH_ERROR.toException("sync load balance unexpected netty "
                    + "error " + "happened, exception: ", e));
        }
        // Closing is this method's own duty and must happen whether or not the sink was ours to
        // fail; the resulting channelInactive covers a sink installed after the claim was taken.
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

        // Check if current HTTP response status is 200; if not 200, indicates abnormal situation, parse chunk and return error
        int httpStatusCode = NettyUtils.getHttpStatusCode(nettyCtx);

        // If status code is not 200, indicates abnormal situation, parse chunk and return error
        if (httpStatusCode != StatusEnum.SUCCESS.getCode()) {
            NettyUtils.cacheBuffer(nettyCtx, obj);
            // Wait for the whole error body before failing; a chunked non-200 response would
            // otherwise be truncated to its first chunk, corrupting upstream business-error JSON.
            if (!(obj instanceof LastHttpContent)) {
                return;
            }
            String body = NettyUtils.readBody(nettyCtx);
            NettyUtils.finishNettyWithException(nettyCtx,
                    new HttpErrorResponseException(httpStatusCode, body));
            return;
        }

        // Add to buffer
        NettyUtils.cacheBuffer(nettyCtx, obj);

        // For non-streaming calls, wait until last chunk arrives before parsing
        if (!(obj instanceof LastHttpContent)) {
            return;
        }

        try {
            byte[] bodyBytes = NettyUtils.getBodyBytes(nettyCtx);
            nettyCtx.getByteDataList().clear();
            nettyCtx.getByteDataSize().reset();

            // For non-streaming calls, ignore if data is empty
            // Normally bodyBytes should not be empty
            // Currently only occurs when engine disconnects abnormally, due to HttpObjectAggregator aggregation, returns a FullResponse with empty body
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
            // Upstream stream complete
            nettyCtx.getSink().complete();
            // Downstream netty complete
            NettyUtils.finish(nettyCtx);

        } catch (FlexLBException e) {
            nettyCtx.getSink().error(e);
            NettyUtils.finish(nettyCtx);
        }
    }
}
