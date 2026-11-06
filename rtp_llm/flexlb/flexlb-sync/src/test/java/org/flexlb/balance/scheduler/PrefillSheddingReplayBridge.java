package org.flexlb.balance.scheduler;

import org.flexlb.config.PrefillQueueSheddingConfig;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;

/** Test-only line protocol: the offline worker model executes the production shedding policies. */
public final class PrefillSheddingReplayBridge {
    public static void main(String[] args) throws Exception {
        var config = new PrefillQueueSheddingConfig();
        config.setExpectedWaitThresholdMs(60_000);
        config.setMaxWaitMs(120_000);
        var budget = new PrefillQueueSheddingConfig.DeadlineBudgetConfig();
        budget.setTokensPerSecond(70_000);
        var legacy = new PrefillQueueShedding.Controller();
        var deadline = new PrefillDeadlineBudget();
        var reader = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] header = line.split(" ");
            if (header[0].equals("RESET")) {
                legacy = new PrefillQueueShedding.Controller();
                deadline = new PrefillDeadlineBudget();
                System.out.println("OK");
                continue;
            }
            long now = Long.parseLong(header[1]);
            int workers = Integer.parseInt(header[2]);
            int saturated = Integer.parseInt(header[3]);
            double outstanding = Double.parseDouble(header[4]);
            int count = Integer.parseInt(header[5]);
            int arrivals = Integer.parseInt(header[6]);
            int coldArrivals = Integer.parseInt(header[7]);
            int timeouts = Integer.parseInt(header[8]);
            for (int i = 0; i < arrivals; i++) { legacy.arrive(); }
            for (int i = 0; i < coldArrivals; i++) { legacy.coldArrival(); }
            for (int i = 0; i < timeouts; i++) { legacy.hardTimeout(now); }
            var oldCandidates = new ArrayList<PrefillQueueShedding.Candidate<Long>>();
            var candidates = new ArrayList<PrefillDeadlineBudget.Candidate<Long>>();
            for (int i = 0; i < count; i++) {
                String[] fields = reader.readLine().split(" ");
                long id = Long.parseLong(fields[0]);
                long input = Long.parseLong(fields[1]);
                long hit = Long.parseLong(fields[2]);
                long age = Long.parseLong(fields[3]);
                oldCandidates.add(new PrefillQueueShedding.Candidate<>(id, 50, id,
                        new PrefillQueueShedding.Estimate(input, hit, (input - hit) / 70, Math.max(1, workers)), age));
                candidates.add(new PrefillDeadlineBudget.Candidate<>(id, 50, id, input - hit, hit == 0,
                        Math.max(0, 100_000 - age)));
            }
            var victims = header[0].equals("budget")
                    ? deadline.victims(candidates, new PrefillDeadlineBudget.Fleet(workers, saturated, outstanding, true), budget, false, now)
                    : PrefillQueueShedding.adaptiveVictims(oldCandidates, config, null, legacy, now, saturated > workers / 2);
            if (!header[0].equals("budget")) { legacy.recordEvictions(now, victims.size(), victims.size()); }
            System.out.println(String.join(",", victims.stream().map(String::valueOf).toList()));
        }
    }
}
