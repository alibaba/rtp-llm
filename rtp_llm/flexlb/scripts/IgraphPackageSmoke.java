import java.util.List;
import org.flexlb.constraint.ConstraintTreeBuilder;
import org.flexlb.constraint.ConstraintTreeCsrCodec;
import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;

/** Standalone local package check: no Spring application startup, discovery or online requests. */
public class IgraphPackageSmoke {
    public static void main(String[] args) throws Exception {
        var bindings = java.util.Collections.list(ClassLoader.getSystemClassLoader()
                .getResources("org/slf4j/impl/StaticLoggerBinder.class"));
        if (bindings.size() != 1 || !org.slf4j.LoggerFactory.getILoggerFactory().getClass().getName()
                .equals("ch.qos.logback.classic.LoggerContext")) {
            throw new AssertionError("packaged dependencies changed Master logging: " + bindings);
        }
        System.out.println("PACK_SINGLE_LOGBACK_BINDING_OK");
        var settings = new org.flexlb.constraint.BucketSidReader.Settings("", 4000, 4, 2000,
                java.time.Duration.ofSeconds(5), java.time.Duration.ofMinutes(5), 0,
                org.flexlb.constraint.BucketSidReader.BucketAlgorithm.ITEM_ID_MOD, 2000);
        if (!settings.key(3999).equals("3999") || org.flexlb.constraint.BucketSidReader.bucketForItem(
                "9223372036854775808", 4000, settings.bucketAlgorithm()) != 3808) {
            throw new AssertionError("packaged numeric bucket contract failed");
        }
        System.out.println("PACK_NUMERIC_BUCKET_OK");
        var skipSettings = new org.flexlb.constraint.BucketSidReader.Settings("", 1, 1, 2000,
                java.time.Duration.ofSeconds(5), java.time.Duration.ofMinutes(5), 0,
                org.flexlb.constraint.BucketSidReader.BucketAlgorithm.ITEM_ID_MOD, 2000,
                org.flexlb.constraint.BucketSidReader.EmptySidPolicy.SKIP);
        var input = new org.flexlb.constraint.BucketSidReader((key, limit, timeout) ->
                java.util.concurrent.CompletableFuture.completedFuture(List.of(
                        new org.flexlb.constraint.source.SidBucketClient.Row(key, "0", ""),
                        new org.flexlb.constraint.source.SidBucketClient.Row(key, "1", "C1C2"))),
                skipSettings).read(() -> true);
        if (input.itemCount() != 2 || input.skippedEmptySids() != 1 || input.eligibleItems() != 1
                || !input.sids().equals(List.of("C1C2"))) {
            throw new AssertionError("packaged empty SID policy failed");
        }
        System.out.println("PACK_EMPTY_SID_POLICY_OK");
        for (String name : List.of("org.flexlb.constraint.BucketSidReader",
                "org.flexlb.constraint.IgraphConstraintTreePoller",
                "org.flexlb.httpserver.IgraphConstraintTreeServer",
                "org.flexlb.igraph.IgraphSidBucketClient",
                "org.flexlb.igraph.IgraphSidConfiguration",
                "com.taobao.igraph.client.core.IGraphClientBuilder",
                "com.taobao.igraph.client.core.IGraphClient",
                "com.taobao.igraph.client.model.AsyncQueryResultContext")) {
            Class<?> type = Class.forName(name);
            System.out.println("PACK_CLASS_OK " + name + " methods=" + type.getDeclaredMethods().length);
        }
        try (ConstraintTreeBuilder builder = new ConstraintTreeBuilder()) {
            var request = new BuildRequest(1, "package_smoke", 1699, 151645, "_",
                    List.of(new int[]{170000, 170001}), null);
            var decoded = ConstraintTreeCsrCodec.decode(ConstraintTreeCsrCodec.encode(builder.build(request)));
            if (decoded.sidCount() != 1) { throw new AssertionError("packaged CSR round-trip failed"); }
            System.out.println("PACK_CSR_ROUNDTRIP_OK");
        }
    }
}
