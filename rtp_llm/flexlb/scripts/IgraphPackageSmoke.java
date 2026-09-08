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
