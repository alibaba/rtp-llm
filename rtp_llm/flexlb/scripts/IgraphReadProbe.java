import java.time.Duration;
import java.util.concurrent.TimeUnit;
import org.flexlb.igraph.IgraphSidConfiguration;
import org.springframework.core.env.StandardEnvironment;

/** Read one explicitly supplied bucket using packaged production SDK/config. Never builds or publishes a tree. */
public class IgraphReadProbe {
    public static void main(String[] args) throws Exception {
        if (args.length != 1 || !args[0].matches("[0-9]+") || Integer.parseInt(args[0]) >= 4000) {
            throw new IllegalArgumentException("usage: IgraphReadProbe.java <decimal bucket 0..3999>");
        }
        var env = new StandardEnvironment();
        var configuration = new IgraphSidConfiguration();
        long started = System.nanoTime();
        var sdk = configuration.constraintTreeIgraphClient(env);
        try {
            var client = configuration.sidBucketClient(sdk, env);
            var timeout = Duration.ofMillis(Long.parseLong(env.getProperty(
                    "constraint.tree.igraph.query.timeout.ms", "5000")));
            var future = client.readAsync(args[0], 2001, timeout);
            try {
                var rows = future.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
                System.out.println("IGRAPH_READ_PROBE bucket=" + args[0] + " rows=" + rows.size()
                        + " elapsedMs=" + TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started)
                        + " (read only; no tree publication; completeness NOT verified)");
                if (rows.size() >= 2000) { throw new IllegalStateException("possible server/index truncation"); }
            } finally { future.cancel(true); }
        } finally { sdk.close(); }
    }
}
