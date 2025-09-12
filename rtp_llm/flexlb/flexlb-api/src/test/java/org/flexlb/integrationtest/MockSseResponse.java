package org.flexlb.integrationtest;

import lombok.Getter;
import lombok.SneakyThrows;
import okhttp3.mockwebserver.MockResponse;
import okio.Buffer;

import java.lang.reflect.Field;

@Getter
public class MockSseResponse {

    private final MockResponse mockResponse = new MockResponse();

    Buffer bodyBuffer = new Buffer();

    public void startStreaming() {
        mockResponse.addHeader("Cache-Control", "no-cache");
        mockResponse.addHeader("Content-Type", "text/event-stream");
        mockResponse.addHeader("Transfer-Encoding", "chunked");
        mockResponse.setResponseCode(200);
        mockResponse.removeHeader("Content-Length");
    }

    public void appendData(String message, boolean isWithBlank) {
        String preFix = isWithBlank ? "data: " : "data:";
        String msgDataByte = preFix + message + "\r\n\r\n";
        Buffer msgDataBuf = new Buffer().writeUtf8(msgDataByte);
        long msgDataBufSize = msgDataBuf.size();

        bodyBuffer.writeHexadecimalUnsignedLong(msgDataBufSize);
        bodyBuffer.writeUtf8("\r\n");
        bodyBuffer.write(msgDataBuf, msgDataBufSize);
        bodyBuffer.writeUtf8("\r\n");
    }

    @SneakyThrows
    public void endStreaming() {

        bodyBuffer.writeUtf8("0\r\n");
        Field body = MockResponse.class.getDeclaredField("body");
        body.setAccessible(true);
        body.set(this.mockResponse, bodyBuffer);
    }


}
