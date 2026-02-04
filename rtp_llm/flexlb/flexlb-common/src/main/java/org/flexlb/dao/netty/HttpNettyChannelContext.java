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
 * NettyChannel 的上下文信息
 *
 * @author lushirong
 */
@Data
@NoArgsConstructor
@Builder
@AllArgsConstructor
public class HttpNettyChannelContext<T> {

    /**
     * 客户自定义请求
     */
    private Object request;
    /**
     * 与模型服务连接的 channel，可用于通信
     */
    private Channel channel;
    /**
     * 流式调用中与用户交互的发射器
     */
    private FluxSink<T> sink;
    /**
     * 请求模型服务的 http 响应
     */
    private HttpResponse httpResp;
    /**
     * 流式调用中，缓存模型服务的 sse 响应，直到一次事件的结束
     */
    private List<Byte> buffer;

    private List<ByteData> byteDataList;

    private LongAdder byteDataSize;

    /**
     * netty 触发 read 事件时调用的回调函数
     */
    private BiConsumer<HttpNettyChannelContext<T>, HttpObject> readCallback;
    /**
     * netty channel 连接断开的回调函数
     */
    private Consumer<HttpNettyChannelContext<T>> channelInactiveCallback;
    /**
     * netty 交互过程中抛出异常时调用的回调函数
     */
    private BiConsumer<HttpNettyChannelContext<T>, Throwable> errorCallback;
    /**
     * netty channel 增强处理回调
     */
    private Consumer<HttpNettyChannelContext<T>> channelEnhanceCallback;
    /**
     * 处理是否结束
     */
    private boolean finish;

    @Data
    @AllArgsConstructor
    public static class ByteData {
        private byte[] data;
    }
}
