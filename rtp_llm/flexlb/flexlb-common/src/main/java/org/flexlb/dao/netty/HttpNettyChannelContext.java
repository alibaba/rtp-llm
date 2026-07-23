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

    /**
     * {@code sink} is installed by the thread that issues the request while the Netty threads may
     * already be ending the exchange, so both fields are confined to this monitor — which also
     * supplies the happens-before that plain fields never had between the installing thread and
     * the Netty ones.
     *
     * <p>The monitor carries a claim, {@link #installSink} plus {@link #claimTermination}, over
     * the paths that end an exchange <em>early</em>: whichever of them runs first, exactly one
     * comes away owning the sink's terminal signal. The claim does not reach the ordinary
     * completion path, which emits onto the sink directly and only then marks {@link #setFinish}
     * (see {@code NettyUtils#finish}). That path is safe for a reason outside this monitor: it
     * runs on the inbound executor and marks the exchange finished before the close it issues can
     * deliver a channelInactive, so every later claimant finds the claim already taken. Move
     * either side off the thread it runs on today and it is that argument, not the monitor, that
     * gives way.
     */
    public synchronized FluxSink<T> getSink() {
        return sink;
    }

    /** Unused today; kept so {@code @Data} cannot put an unsynchronized setter back in its place. */
    public synchronized void setSink(FluxSink<T> sink) {
        this.sink = sink;
    }

    public synchronized boolean isFinish() {
        return finish;
    }

    public synchronized void setFinish(boolean finish) {
        this.finish = finish;
    }

    /**
     * Installs the sink and reports whether the exchange had already ended before it existed, in
     * which case the caller owns terminating it — whoever ended the exchange found no sink to end.
     */
    public synchronized boolean installSink(FluxSink<T> sink) {
        this.sink = sink;
        return finish;
    }

    /**
     * Claims the exclusive right to terminate the sink and returns it. Null means the caller does
     * not own the termination: either the exchange has already ended, or the sink does not exist
     * yet and the thread installing it terminates instead.
     */
    public synchronized FluxSink<T> claimTermination() {
        if (finish) {
            return null;
        }
        finish = true;
        return sink;
    }

    @Data
    @AllArgsConstructor
    public static class ByteData {
        private byte[] data;
    }
}
