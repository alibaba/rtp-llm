package org.flexlb.transport;

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.socket.SocketChannel;
import io.netty.handler.codec.http.HttpObject;
import io.netty.util.AttributeKey;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.netty.HttpNettyChannelContext;

/**
 * @author lushirong
 */
@SuppressWarnings({"rawtypes", "unchecked"})
@ChannelHandler.Sharable
@Slf4j
public class HttpNettyClientHandler extends SimpleChannelInboundHandler<HttpObject> {
    public static final String CHANNEL_CONTEXT_ATTR_NAME = "nettyCtx";
    private final Bootstrap bootstrap;

    public HttpNettyClientHandler(Bootstrap bootstrap) {
        this.bootstrap = bootstrap;
    }

    /**
     * Initiate a connection via Netty
     *
     * @param host host
     * @param port port
     * @return future representing connection completion
     */
    public ChannelFuture connect(String host, int port) {
        return bootstrap.connect(host, port);
    }

    public HttpNettyChannelContext getNettyChannelContext(Channel channel) {
        return (HttpNettyChannelContext) channel.attr(AttributeKey.valueOf(CHANNEL_CONTEXT_ATTR_NAME)).get();
    }

    public void setNettyChannelContext(Channel channel, HttpNettyChannelContext nettyCtx) {
        channel.attr(AttributeKey.valueOf(CHANNEL_CONTEXT_ATTR_NAME)).set(nettyCtx);
    }

    /**
     * Triggered when channel is closed
     */
    @Override
    public void channelInactive(ChannelHandlerContext ctx) {
        try {
            HttpNettyChannelContext nettyCtx = getNettyChannelContext(ctx.channel());
            if (nettyCtx != null && nettyCtx.getChannelInactiveCallback() != null) {
                nettyCtx.getChannelInactiveCallback().accept(nettyCtx);
            }
        } catch (Throwable t) {
            log.error("channelInactive exceptionCaught", t);
        }
    }

    @Override
    protected void channelRead0(ChannelHandlerContext channelHandlerContext, HttpObject obj) {
        try {
            HttpNettyChannelContext nettyCtx = getNettyChannelContext(channelHandlerContext.channel());
            if (nettyCtx != null && nettyCtx.getReadCallback() != null) {
                nettyCtx.getReadCallback().accept(nettyCtx, obj);
            }
        } catch (Throwable t) {
            log.error("channelRead0 exceptionCaught", t);
        }
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext channelHandlerContext, Throwable cause) {
        try {
            HttpNettyChannelContext nettyCtx = getNettyChannelContext(channelHandlerContext.channel());
            if (nettyCtx != null && nettyCtx.getErrorCallback() != null) {
                nettyCtx.getErrorCallback().accept(nettyCtx, cause);
            }
        } catch (Throwable t) {
            log.error("exceptionCaught: ", t);
        }
    }

    public void channelEnhance(SocketChannel channel) {
        try {
            HttpNettyChannelContext nettyCtx = getNettyChannelContext(channel);
            if (nettyCtx != null && nettyCtx.getChannelEnhanceCallback() != null) {
                nettyCtx.getChannelEnhanceCallback().accept(nettyCtx);
            }
        } catch (Throwable t) {
            log.error("enhanceChannelPipeline exceptionCaught", t);
        }
    }
}