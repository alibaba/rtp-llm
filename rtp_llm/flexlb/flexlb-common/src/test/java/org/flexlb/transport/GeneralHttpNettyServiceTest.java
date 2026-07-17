package org.flexlb.transport;

import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelFuture;
import io.netty.channel.embedded.EmbeddedChannel;
import io.netty.handler.codec.http.DefaultHttpContent;
import io.netty.handler.codec.http.DefaultHttpResponse;
import io.netty.handler.codec.http.DefaultLastHttpContent;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import org.flexlb.exception.EngineAbnormalDisconnectException;
import org.flexlb.exception.HttpErrorResponseException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Drives the shared HTTP transport through a Netty {@link EmbeddedChannel} — no sockets — to pin
 * the aggregation contract both the 200 and non-200 paths rely on: the sink terminates on
 * {@code LastHttpContent} with the full multi-chunk body, and a peer that disconnects before
 * {@code LastHttpContent} terminates the sink with an error instead of leaving the caller waiting
 * forever (the read-timeout handler stops firing once the channel is inactive).
 */
class GeneralHttpNettyServiceTest {

    /** Deserialization target for the 200 path. */
    static class EchoResponse {
        public String message;
    }

    private EmbeddedChannel channel;
    private GeneralHttpNettyService service;

    @BeforeEach
    void setUp() {
        HttpNettyClientHandler handler = new HttpNettyClientHandler(new Bootstrap()) {
            @Override
            public ChannelFuture connect(String host, int port) {
                return channel.newSucceededFuture();
            }
        };
        channel = new EmbeddedChannel(handler);
        service = new GeneralHttpNettyService(handler);
    }

    @AfterEach
    void tearDown() {
        channel.finishAndReleaseAll();
    }

    private CompletableFuture<EchoResponse> sendRequest() throws Exception {
        CompletableFuture<EchoResponse> result = service
                .request(Map.of("k", "v"), URI.create("http://backend:8080"), "/path", EchoResponse.class)
                .toFuture();
        // The request is written from the shared request-thread scheduler; pump the embedded
        // event loop until the outbound write lands so inbound frames cannot race the sink setup.
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (channel.outboundMessages().isEmpty()) {
            channel.runPendingTasks();
            if (System.nanoTime() > deadline) {
                throw new AssertionError("request was never written to the channel");
            }
            Thread.sleep(1);
        }
        return result;
    }

    private DefaultHttpContent content(String text) {
        return new DefaultHttpContent(Unpooled.copiedBuffer(text, StandardCharsets.UTF_8));
    }

    @Test
    void non200ResponseSplitAcrossChunksFailsWithTheFullConcatenatedBody() throws Exception {
        CompletableFuture<EchoResponse> result = sendRequest();

        channel.writeInbound(new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.BAD_REQUEST));
        channel.writeInbound(content("{\"error_code\":400,"));
        channel.writeInbound(content("\"error_message\":"));
        channel.writeInbound(new DefaultLastHttpContent(
                Unpooled.copiedBuffer("\"bad batch_count\"}", StandardCharsets.UTF_8)));

        ExecutionException e = assertThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));
        HttpErrorResponseException httpError = assertInstanceOf(HttpErrorResponseException.class, e.getCause(),
                "a completed non-200 exchange must surface as HttpErrorResponseException, not a transport error");
        assertEquals(400, httpError.getStatusCode());
        assertEquals("{\"error_code\":400,\"error_message\":\"bad batch_count\"}", httpError.getBody(),
                "the error body must be the concatenation of every chunk, not just the first one");
    }

    @Test
    void okResponseSplitAcrossChunksParsesTheAggregatedBody() throws Exception {
        CompletableFuture<EchoResponse> result = sendRequest();

        channel.writeInbound(new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK));
        channel.writeInbound(content("{\"message\":"));
        channel.writeInbound(new DefaultLastHttpContent(
                Unpooled.copiedBuffer("\"hello\"}", StandardCharsets.UTF_8)));

        EchoResponse response = result.get(5, TimeUnit.SECONDS);
        assertNotNull(response);
        assertEquals("hello", response.message);
    }

    @Test
    void peerDisconnectMidNon200BodyErrorsTheSinkInsteadOfHangingForever() throws Exception {
        CompletableFuture<EchoResponse> result = sendRequest();

        channel.writeInbound(new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.BAD_GATEWAY));
        channel.writeInbound(content("{\"partial\":"));
        channel.close();

        ExecutionException e = assertThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));
        assertInstanceOf(EngineAbnormalDisconnectException.class, e.getCause(),
                "the non-200 path waits for LastHttpContent, so a mid-body disconnect must fail the "
                        + "sink — nothing else ever terminates it once the channel is gone");
    }

    @Test
    void peerDisconnectMid200BodyErrorsTheSinkInsteadOfHangingForever() throws Exception {
        CompletableFuture<EchoResponse> result = sendRequest();

        channel.writeInbound(new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK));
        channel.writeInbound(content("{\"message\":"));
        channel.close();

        ExecutionException e = assertThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));
        assertInstanceOf(EngineAbnormalDisconnectException.class, e.getCause(),
                "the 200 aggregation path has the same LastHttpContent dependency as the non-200 path");
    }

    @Test
    void completedExchangeIsNotDisturbedByTheServiceClosingItsOwnChannel() throws Exception {
        CompletableFuture<EchoResponse> result = sendRequest();

        channel.writeInbound(new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK));
        channel.writeInbound(new DefaultLastHttpContent(
                Unpooled.copiedBuffer("{\"message\":\"done\"}", StandardCharsets.UTF_8)));

        // The service closes the channel itself right after completing the sink, which fires the
        // same channelInactive the disconnect guard listens on — a completed exchange must come
        // out of that as a success, not be retroactively failed.
        assertEquals("done", result.get(5, TimeUnit.SECONDS).message);
        assertTrue(result.isDone());
        assertEquals("done", result.get().message);
    }
}
