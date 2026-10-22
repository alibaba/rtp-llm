import org.flexlb.balance.scheduler.PrefillTrafficTrend;
import org.flexlb.config.PrefillQueueSheddingConfig;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/** Replay time_ms,arrivals,input_tokens,queued_work_ms,discarded_work_ms CSV through the live detector.
 * Counts are interval deltas; queue is the current total in fleet-service milliseconds.
 * Usage: java -cp flexlb-sync/target/classes:flexlb-common/target/classes tools/PrefillTrafficTrendReplay.java trace.csv
 */
public final class PrefillTrafficTrendReplay {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) { throw new IllegalArgumentException("expected CSV path"); }
        var config = new PrefillQueueSheddingConfig.RecoveryConfig();
        var trend = new PrefillTrafficTrend(config);
        try (var input = Files.newBufferedReader(Path.of(args[0]))) {
            if (!"time_ms,arrivals,input_tokens,queued_work_ms,discarded_work_ms".equals(input.readLine())) {
                throw new IllegalArgumentException("unexpected CSV header");
            }
            System.out.println("time_ms,state,arrival_qps,previous_qps,input_tps,previous_input_tps,queued_work_ms,grace,remaining_ms");
            for (String line; (line = input.readLine()) != null;) {
                if (line.isBlank()) { continue; }
                String[] values = line.split(",", -1);
                if (values.length != 5) { throw new IllegalArgumentException("expected five columns"); }
                long at = Long.parseLong(values[0]), count = Long.parseLong(values[1]), tokens = Long.parseLong(values[2]);
                if (count < 0 || tokens < 0 || count == 0 && tokens != 0) { throw new IllegalArgumentException("invalid traffic counts"); }
                for (long i = 0; i < count; i++) { trend.arrive(tokens / count + (i < tokens % count ? 1 : 0)); }
                trend.discard(Double.parseDouble(values[4]));
                var s = trend.observe(at, Double.parseDouble(values[3]), 30000, 60000);
                System.out.printf(Locale.ROOT, "%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%d%n",
                        at, s.state(), s.arrivalQps(), s.previousQps(), s.inputTps(), s.previousInputTps(),
                        s.queuedWorkMs(), s.grace(), s.graceRemainingMs());
            }
        }
    }
}
