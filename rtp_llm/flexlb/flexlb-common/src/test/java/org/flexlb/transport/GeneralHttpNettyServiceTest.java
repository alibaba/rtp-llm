package org.flexlb.transport;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.util.concurrent.DefaultEventExecutorGroup;
import io.netty.util.concurrent.EventExecutorGroup;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GeneralHttpNettyServiceTest {

    private final EventLoopGroup eventLoopGroup = new NioEventLoopGroup(1);
    private final EventExecutorGroup callbackExecutor = new DefaultEventExecutorGroup(1);
    private final AtomicReference<byte[]> receivedBody = new AtomicReference<>();
    private HttpServer server;
    private GeneralHttpNettyService httpService;
    private URI serverUri;

    @BeforeEach
    void setUp() throws IOException {
        Bootstrap bootstrap = new Bootstrap();
        HttpNettyClientHandler clientHandler = new HttpNettyClientHandler(bootstrap);
        bootstrap.group(eventLoopGroup)
                .channel(NioSocketChannel.class)
                .handler(new ChannelInitializer<SocketChannel>() {
                    @Override
                    protected void initChannel(SocketChannel channel) {
                        channel.pipeline()
                                .addLast("codec", new HttpClientCodec())
                                .addLast(callbackExecutor, "handler", clientHandler);
                    }
                });
        httpService = new GeneralHttpNettyService(clientHandler);

        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/update_constraint_tree", this::handleUpdate);
        server.createContext("/constraint_tree_status", this::handleStatus);
        server.start();
        serverUri = URI.create("http://127.0.0.1:" + server.getAddress().getPort());
    }

    @AfterEach
    void tearDown() {
        if (server != null) {
            server.stop(0);
        }
        callbackExecutor.shutdownGracefully().syncUninterruptibly();
        eventLoopGroup.shutdownGracefully().syncUninterruptibly();
    }

    @Test
    void sendsPreSerializedJsonBytesAndParsesResponse() {
        byte[] payload = "{\"version\":202609031210,\"prefix_dict\":{\"1699\":[169967]}}"
                .getBytes(StandardCharsets.UTF_8);

        TestResponse response = httpService.requestRawJson(
                        payload, serverUri, "/update_constraint_tree", TestResponse.class)
                .block(Duration.ofSeconds(5));

        assertArrayEquals(payload, receivedBody.get());
        assertEquals("accepted", response.status());
        assertEquals(202609031210L, response.requestedVersion());
    }

    @Test
    void performsBodylessGetAndParsesResponse() {
        TestResponse response = httpService.get(
                        serverUri, "/constraint_tree_status", TestResponse.class)
                .block(Duration.ofSeconds(5));

        assertEquals("ready", response.status());
        assertEquals(202609031210L, response.version());
    }

    private void handleUpdate(HttpExchange exchange) throws IOException {
        assertEquals("POST", exchange.getRequestMethod());
        receivedBody.set(exchange.getRequestBody().readAllBytes());
        respond(exchange, "{\"status\":\"accepted\",\"version\":0,"
                + "\"requested_version\":202609031210}");
    }

    private void handleStatus(HttpExchange exchange) throws IOException {
        assertEquals("GET", exchange.getRequestMethod());
        assertEquals(0, exchange.getRequestBody().readAllBytes().length);
        respond(exchange, "{\"status\":\"ready\",\"version\":202609031210,"
                + "\"requested_version\":202609031210}");
    }

    private void respond(HttpExchange exchange, String json) throws IOException {
        byte[] body = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, body.length);
        exchange.getResponseBody().write(body);
        exchange.close();
    }

    private record TestResponse(
            String status,
            long version,
            @com.fasterxml.jackson.annotation.JsonProperty("requested_version") long requestedVersion) {
    }
}
