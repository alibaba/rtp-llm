package org.flexlb.transport;

import com.taobao.eagleeye.EagleEye;
import com.taobao.eagleeye.RpcContext_inner;
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

import static com.taobao.eagleeye.EagleEye.TYPE_HSF_CLIENT;

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
     * 由 netty 发起一个连接
     * @param host host
     * @param port port
     * @return 表示连接是否完成的未来
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
     * 在通道被关闭的时候触发
     */
    @Override
    public void channelInactive(ChannelHandlerContext ctx) {
        RpcContext_inner preRpcContext = EagleEye.getRpcContext();
        try {
            HttpNettyChannelContext nettyCtx = getNettyChannelContext(ctx.channel());
            if (nettyCtx != null && nettyCtx.getChannelInactiveCallback() != null) {
                if (nettyCtx.getRequestCtx() != null) {
                    EagleEye.setRpcContext(nettyCtx.getRequestCtx().getEagleTraceCtx());
                }
                nettyCtx.getChannelInactiveCallback().accept(nettyCtx);
            }
        } catch (Throwable t) {
            log.error("channelInactive exceptionCaught", t);
        } finally {
            EagleEye.setRpcContext(preRpcContext);
        }
    }

    @Override
    protected void channelRead0(ChannelHandlerContext channelHandlerContext, HttpObject obj) {
        RpcContext_inner preRpcContext = EagleEye.getRpcContext();
        try {
            HttpNettyChannelContext nettyCtx = getNettyChannelContext(channelHandlerContext.channel());
            if (nettyCtx != null && nettyCtx.getReadCallback() != null) {
                if (nettyCtx.getRequestCtx() != null) {
                    EagleEye.setRpcContext(nettyCtx.getRequestCtx().getEagleTraceCtx());
                    EagleEye.rpcClientRecv(EagleEye.RPC_RESULT_SUCCESS, TYPE_HSF_CLIENT);
                }
                nettyCtx.getReadCallback().accept(nettyCtx, obj);
            }
        } catch (Throwable t) {
            log.error("channelRead0 exceptionCaught", t);
        } finally {
            EagleEye.setRpcContext(preRpcContext);
        }
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext channelHandlerContext, Throwable cause) {
        RpcContext_inner preRpcContext = EagleEye.getRpcContext();
        try {
            HttpNettyChannelContext nettyCtx = getNettyChannelContext(channelHandlerContext.channel());
            if (nettyCtx != null && nettyCtx.getErrorCallback() != null) {
                if (nettyCtx.getRequestCtx() != null) {
                    EagleEye.setRpcContext(nettyCtx.getRequestCtx().getEagleTraceCtx());
                    EagleEye.rpcClientRecv(EagleEye.RPC_RESULT_RPC_ERROR, TYPE_HSF_CLIENT);
                }
                nettyCtx.getErrorCallback().accept(nettyCtx, cause);
            }
        } catch (Throwable t) {
            log.error("exceptionCaught: ", t);
        } finally {
            EagleEye.setRpcContext(preRpcContext);
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
