package org.flexlb.util;

import io.netty.buffer.ByteBuf;
import io.netty.handler.codec.http.HttpContent;
import io.netty.handler.codec.http.HttpObject;
import io.netty.handler.codec.http.HttpResponse;
import io.netty.handler.codec.http.HttpResponseStatus;
import org.flexlb.dao.netty.HttpNettyChannelContext;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Optional;

public class NettyUtils {

    public static <T> void cacheBuffer(HttpNettyChannelContext<T> nettyCtx, HttpObject obj) {

        ByteBuf content = ((HttpContent) obj).content();
        nettyCtx.getByteDataSize().add(content.readableBytes());
        byte[] buffer = new byte[content.readableBytes()];
        content.getBytes(content.readerIndex(), buffer);

        List<HttpNettyChannelContext.ByteData> byteDataList = nettyCtx.getByteDataList();
        byteDataList.add(new HttpNettyChannelContext.ByteData(buffer));
    }

    public static <T> String readBody(HttpNettyChannelContext<T> nettyCtx) {
        byte[] mergedData = getBodyBytes(nettyCtx);
        return new String(mergedData, StandardCharsets.UTF_8);
    }

    public static <T> byte[] getBodyBytes(HttpNettyChannelContext<T> nettyCtx) {
        List<HttpNettyChannelContext.ByteData> byteDataList = nettyCtx.getByteDataList();
        // 如果只有一个 chunk, 那么不需要合并, 直接返回
        if (byteDataList.size() == 1) {
            return byteDataList.getFirst().getData();
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

    public static <T> void finishNettyWithException(HttpNettyChannelContext<T> nettyCtx, Throwable e) {
        finish(nettyCtx);
        nettyCtx.getSink().error(e);
    }

    public static <T> void finish(HttpNettyChannelContext<T> nettyCtx) {
        nettyCtx.setFinish(true);
        nettyCtx.getChannel().close();
    }

    public static <T> int getHttpStatusCode(HttpNettyChannelContext<T> nettyCtx) {
        //noinspection deprecation
        return Optional.of(nettyCtx)
                .map(HttpNettyChannelContext::getHttpResp)
                .map(HttpResponse::getStatus)
                .map(HttpResponseStatus::code)
                .orElse(-99);
    }
}
