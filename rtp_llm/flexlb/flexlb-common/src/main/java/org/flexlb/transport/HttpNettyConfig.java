package org.flexlb.transport;

import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.util.concurrent.DefaultEventExecutorGroup;
import io.netty.util.concurrent.DefaultThreadFactory;
import io.netty.util.concurrent.EventExecutorGroup;
import io.netty.util.concurrent.RejectedExecutionHandlers;
import org.flexlb.constant.CommonConstants;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.TimeUnit;

/**
 * @author lushirong
 */
@Configuration
public class HttpNettyConfig {

    private final int responseTimeoutSeconds = 120;
    private final int nettyMaxChunkSize = 8092;
    private final int eventExecuteThreads = 10 * Runtime.getRuntime().availableProcessors();

    @Bean(name = "nettyClient")
    public HttpNettyClientHandler createNettyClientHandler() {
        Bootstrap bootstrap = new Bootstrap();
        EventLoopGroup group = new NioEventLoopGroup(Runtime.getRuntime().availableProcessors());
        HttpNettyClientHandler handler = new HttpNettyClientHandler(bootstrap);
        EventExecutorGroup defaultEventExecutorGroup = new DefaultEventExecutorGroup(eventExecuteThreads,
                new DefaultThreadFactory("default-custom-executor"),
                // 使用有界队列，容量为1000
                1000, RejectedExecutionHandlers.reject());
        int requestTimeoutMillis = 500;
        bootstrap.group(group)
                .channel(NioSocketChannel.class)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, requestTimeoutMillis)
                .option(ChannelOption.TCP_NODELAY, true)
                .option(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
                .handler(new ChannelInitializer<SocketChannel>() {
                    @Override
                    protected void initChannel(SocketChannel ch) {
                        // pipeline里面的ChannelHandler顺序很重要
                        ch.pipeline()
                                .addLast(CommonConstants.CODEC, new HttpClientCodec(8192, 8192, nettyMaxChunkSize))
                                .addLast("timeoutHandler", new ReadTimeoutHandler(responseTimeoutSeconds,
                                        TimeUnit.SECONDS))
                                .addLast(defaultEventExecutorGroup, "inboundHandler", handler);
                        handler.channelEnhance(ch);
                    }
                });
        return handler;
    }

    /**
     * 同步集群的netty配置
     * 连接超时：300ms, 读超时：2s
     */
    @Bean(name = "syncNettyClient")
    public HttpNettyClientHandler createSyncNettyClientHandler() {
        Bootstrap bootstrap = new Bootstrap();
        EventLoopGroup group = new NioEventLoopGroup(Runtime.getRuntime().availableProcessors());
        HttpNettyClientHandler handler = new HttpNettyClientHandler(bootstrap);
        EventExecutorGroup defaultEventExecutorGroup = new DefaultEventExecutorGroup(eventExecuteThreads,
                new DefaultThreadFactory("default-custom-executor-sync"),
                // 使用有界队列，容量为1000
                1000, RejectedExecutionHandlers.reject());
        bootstrap.group(group)
                .channel(NioSocketChannel.class)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 1000)
                .option(ChannelOption.TCP_NODELAY, true)
                .option(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
                .handler(new ChannelInitializer<SocketChannel>() {
                    @Override
                    protected void initChannel(SocketChannel ch) {
                        // pipeline里面的ChannelHandler顺序很重要
                        ch.pipeline()
                                .addLast(CommonConstants.CODEC, new HttpClientCodec(8192, 8192, nettyMaxChunkSize))
                                .addLast(CommonConstants.TIMEOUT_HANDLER, new ReadTimeoutHandler(3, TimeUnit.SECONDS))
                                .addLast(defaultEventExecutorGroup, "inboundHandler", handler);
                        handler.channelEnhance(ch);
                    }
                });
        return handler;
    }
}