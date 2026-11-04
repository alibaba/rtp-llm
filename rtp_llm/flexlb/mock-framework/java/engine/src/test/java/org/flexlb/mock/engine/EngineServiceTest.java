package org.flexlb.mock.engine;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import java.nio.file.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;

import org.junit.jupiter.api.Test;

class EngineServiceTest {
    static Map<String,Object> configuration(String role) {
        return Map.of("id",role+"-0","component_role",role,"component",Map.of("engine","rtp","config",Map.of("seq_size_per_block",16,
            "max_seq_len",4096,"test_block_num",9,"reserve_block_ratio",0,"mock.compute",Map.of("forward",Map.of("expression","1 + batchSize")))),"environment",Map.of());
    }
    @Test void urlOnlyFrontendRequestDownloadsRealImageAndCompletesPrefillDecode() throws Exception {
        var json = new ObjectMapper();
        var png = new java.io.ByteArrayOutputStream();
        javax.imageio.ImageIO.write(new java.awt.image.BufferedImage(56, 28, java.awt.image.BufferedImage.TYPE_INT_RGB), "png", png);
        var downloads = new java.util.concurrent.atomic.AtomicInteger();
        var images = com.sun.net.httpserver.HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        images.createContext("/image", exchange -> {
            try (exchange) {
                downloads.incrementAndGet(); byte[] bytes = png.toByteArray();
                exchange.sendResponseHeaders(200, bytes.length); exchange.getResponseBody().write(bytes);
            }
        }); images.start();
        var pConfig = json.<ObjectNode>valueToTree(configuration("prefill"));
        var dConfig = json.<ObjectNode>valueToTree(configuration("decode"));
        for (var root : List.of(pConfig, dConfig)) ((ObjectNode) root.path("component").path("config"))
            .put("image_token_id", 1000).put("seq_size_per_block", 2).put("reuse_cache", true);
        var vConfig = org.flexlb.mock.engine.vit.VitConfig.load(json.valueToTree(Map.of("id", "vit-0", "component_role", "vit",
            "component", Map.of("config", Map.of("patch_size", 14, "downsample_ratio", 3, "min_pixels", 0, "max_image_tokens", 1024, "hidden_size", 4, "vit_concurrency", 2,
                "mock.compute.forward.expression", "1")))));
        try (var p = org.flexlb.mock.engine.execution.RtpNode.start(org.flexlb.mock.engine.config.RtpConfig.load(pConfig));
             var d = org.flexlb.mock.engine.execution.RtpNode.start(org.flexlb.mock.engine.config.RtpConfig.load(dConfig));
             var v = org.flexlb.mock.engine.vit.VitNode.start(vConfig, 0, 0)) {
            int dPort = Integer.parseInt(d.rpcAddress().split(":")[1]);
            int vPort = URI.create("http://" + v.endpoints().get("rpc")).getPort();
            var request = GenerateInputPB.newBuilder().setRequestId(123).addAllTokenIds(List.of(7, 1000, 8))
                .addMultimodalInputs(MultimodalInputPB.newBuilder().setMultimodalType(1)
                    .setMultimodalUrl("http://127.0.0.1:" + images.getAddress().getPort() + "/image"))
                .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(3).setIgnoreEos(true).setReuseCache(true)
                    .setCanUsePdSeparation(true).setTimeoutMs(5000).setReturnIncremental(true)
                    .addRoleAddrs(RoleAddrPB.newBuilder().setRole(RoleAddrPB.RoleType.DECODE).setIp("127.0.0.1").setGrpcPort(dPort))
                    .addRoleAddrs(RoleAddrPB.newBuilder().setRole(RoleAddrPB.RoleType.VIT).setIp("127.0.0.1").setGrpcPort(vPort))).build();
            var channel = ManagedChannelBuilder.forTarget(p.rpcAddress()).usePlaintext().build();
            try {
                var replies = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(8, TimeUnit.SECONDS).generateStreamCall(request);
                GenerateOutputsPB last = null;
                while (replies.hasNext()) {
                    last = replies.next(); assertEquals(7, last.getFlattenOutput().getAuxInfo(0).getInputLen());
                }
                assertNotNull(last); assertTrue(last.getFlattenOutput().getFinished(0));
                assertEquals(3, last.getFlattenOutput().getAuxInfo(0).getOutputLen());
                assertEquals(1, downloads.get());
            } finally { channel.shutdownNow().awaitTermination(2, TimeUnit.SECONDS); }
        } finally { images.stop(0); }
    }

}
