package org.flexlb.httpserver;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

@Component
public class FlexlbGrpcServer {

    /**
     * Offset from HTTP port to gRPC port for FlexLB's own servers.
     * This is separate from CommonConstants.GRPC_PORT_OFFSET which applies
     * to backend inference engine ports (HTTP+1→gRPC).
     */
    static final int FLEXLB_GRPC_PORT_OFFSET = 2;
    private static final int DEFAULT_HTTP_PORT = 7001;

    private final FlexlbServiceImpl flexlbServiceImpl;
    private final ConfigService configService;
    private final EventLoopGroup workerGroup;
    private final Environment environment;
    private Server server;
    private NioEventLoopGroup bossGroup;

    public FlexlbGrpcServer(FlexlbServiceImpl flexlbServiceImpl,
                            ConfigService configService,
                            @Qualifier("managedChannelEventLoopGroup") EventLoopGroup workerGroup,
                            Environment environment) {
        this.flexlbServiceImpl = flexlbServiceImpl;
        this.configService = configService;
        this.workerGroup = workerGroup;
        this.environment = environment;
    }

    @PostConstruct
    public void start() throws IOException {
        // Always derive gRPC port from HTTP port.
        // server.port may come from --server.port CLI arg (Spring Environment only)
        // or from -Dserver.port JVM property; check both.
        String portStr = environment.getProperty("server.port");
        if (portStr == null) {
            portStr = System.getProperty("server.port", String.valueOf(DEFAULT_HTTP_PORT));
        }
        int httpPort = Integer.parseInt(portStr);
        int port = httpPort + FLEXLB_GRPC_PORT_OFFSET;

        this.bossGroup = new NioEventLoopGroup(1);

        server = NettyServerBuilder.forPort(port)
                .channelType(NioServerSocketChannel.class)
                .bossEventLoopGroup(bossGroup)
                .workerEventLoopGroup(workerGroup)
                .addService(flexlbServiceImpl)
                .maxInboundMessageSize(16 * 1024 * 1024)
                .build()
                .start();

        Logger.info("FlexLB gRPC server started on port {}", port);
    }

    @PreDestroy
    public void shutdown() {
        if (server != null) {
            server.shutdown();
            try {
                server.awaitTermination(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                server.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        if (bossGroup != null) {
            bossGroup.shutdownGracefully();
        }
    }
}
