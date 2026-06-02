package org.flexlb.dao.netty;

import io.netty.channel.Channel;
import io.netty.handler.codec.http.HttpObject;
import io.netty.handler.codec.http.HttpResponse;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import reactor.core.publisher.FluxSink;

import java.util.List;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * NettyChannel context information
 *
 * @author lushirong
 */
@Data
@NoArgsConstructor
@Builder
@AllArgsConstructor
public class HttpNettyChannelContext<T> {

    /**
     * Client custom request
     */
    private Object request;
    /**
     * Channel connected to model service, used for communication
     */
    private Channel channel;
    /**
     * Emitter for interaction with user in streaming calls
     */
    private FluxSink<T> sink;
    /**
     * HTTP response from model service
     */
    private HttpResponse httpResp;
    /**
     * Cache model service SSE response in streaming calls until end of event
     */
    private List<Byte> buffer;

    private List<ByteData> byteDataList;

    private LongAdder byteDataSize;

    /**
     * Callback function invoked when netty triggers read event
     */
    private BiConsumer<HttpNettyChannelContext<T>, HttpObject> readCallback;
    /**
     * Callback function invoked when netty channel disconnects
     */
    private Consumer<HttpNettyChannelContext<T>> channelInactiveCallback;
    /**
     * Callback function invoked when exception is thrown during netty interaction
     */
    private BiConsumer<HttpNettyChannelContext<T>, Throwable> errorCallback;
    /**
     * Netty channel enhancement processing callback
     */
    private Consumer<HttpNettyChannelContext<T>> channelEnhanceCallback;
    /**
     * Processing completion flag
     */
    private boolean finish;

    @Data
    @AllArgsConstructor
    public static class ByteData {
        private byte[] data;
    }
}
