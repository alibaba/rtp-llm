package org.flexlb.mockengine;

import org.springframework.test.util.ReflectionTestUtils;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * An executable service comparison: same immutable stream, real per-worker cache, serial compute
 * lanes.
 */
class ServiceControl {
    static final int N = 30, S = 270, Q = 240, END = 240_000, DEADLINE = 30_000, BATCH = 32;
    static final long NEVER = Long.MAX_VALUE;

    record Job(int id, int session, int priority, int received, int deadline, List<Long> keys) {
        int input() {
            return keys.size() * 512;
        }

        List<Long> reusableKeys() {
            // These synthetic prompts end on a block boundary. Keep the final token for forward.
            return keys.subList(0, Math.max(0, keys.size() - 1));
        }
    }

    record Work(Job job, MockLruBlockCache.BlockLease lease, int hit, int fresh) {}

    static class Stat {
        long offered, input, completed, goodTokens, fresh, rejected, cached, missedDeadline;
        final List<Integer> lengths = new ArrayList<>();

        void complete(Work work) {
            completed++;
            goodTokens += work.job.input();
            fresh += work.fresh;
            cached += work.hit;
            lengths.add(work.job.input());
        }
    }

    static class Worker {
        final MockLruBlockCache cache = new MockLruBlockCache(119580, 0);

        @SuppressWarnings("unchecked")
        final Map<Long, Integer> keys =
                (Map<Long, Integer>) ReflectionTestUtils.getField(cache, "blocks");

        final List<Job> queue = new ArrayList<>();
        final List<Work> active = new ArrayList<>();
        long due = NEVER;

        int hit(Job job) {
            int n = 0;
            for (long k : job.reusableKeys()) {
                if (!keys.containsKey(k)) break;
                n++;
            }
            return n * 512;
        }
    }

    final Worker[] workers = new Worker[N];
    final Map<Integer, Stat> byPriority = new TreeMap<>();
    final Stat tail = new Stat(), all = new Stat();
    final boolean benefit, onePerBranch, overlap;
    int next;
    final Deque<Job> deadlines = new ArrayDeque<>();
    final Set<Integer> waitingIds = new HashSet<>();
    long digest = 1;
    final PrefillRecoveryReplay gen = new PrefillRecoveryReplay();

    final int rate, batchLimit;

    ServiceControl(String image, String policy, int rate) throws Exception {
        this.rate = rate;
        batchLimit =
                policy.contains("cap")
                        ? Integer.parseInt(policy.substring(policy.lastIndexOf("cap") + 3))
                        : BATCH;
        benefit = policy.contains("benefit");
        onePerBranch = policy.contains("one");
        overlap = policy.contains("overlap");
        if (PrefillRecoveryReplay.P != N
                || PrefillRecoveryReplay.S != S
                || PrefillRecoveryReplay.normal != Q) throw new AssertionError("stream mismatch");
        for (int s = 0; s < S; s++) {
            gen.turns[s] = s * PrefillRecoveryReplay.rotate / S + 4096;
            gen.sessions.add(s);
        }

        for (int i = 0; i < gen.resumeId; i++) {
            int s = gen.advanceSession(i);
            gen.turns[s]++;
        }
        try (var in =
                new DataInputStream(
                        new BufferedInputStream(Files.newInputStream(Path.of(image))))) {
            if (in.readInt() != N) throw new AssertionError("worker count");
            for (int i = 0; i < N; i++) {
                Worker w = workers[i] = new Worker();
                int n = in.readInt();
                for (int k = 0; k < n; k++) w.keys.put(in.readLong(), 0);
                if (w.keys.size() != n) throw new AssertionError("restore");
            }
            if (in.read() != -1) throw new AssertionError();
        }
    }

    static long price(int tokens) {
        return Math.max(1L, (long) (10 + 0.0125 * tokens));
    }

    long arrival(int i) {
        return (long) i * 1000 / rate;
    }

    long stop() {
        return (long) END * Q / rate;
    }

    boolean measured(long t) {
        return t >= stop() / 2 && t < stop();
    }

    void enqueue(int t) throws Exception {
        int id = gen.resumeId + next++, s = gen.advanceSession(id), turn = gen.turns[s]++;
        var keys = PrefillRecoveryReplay.sessionKeys(s, turn, id);
        int p =
                switch (id % 20) {
                    case 0 -> 39;
                    case 1, 2, 3 -> 40;
                    case 4, 5 -> 49;
                    default -> 50;
                };
        var j = new Job(id, s, p, t, t + DEADLINE, keys);
        workers[s % N].queue.add(j);
        deadlines.addLast(j);
        waitingIds.add(j.id);
        digest = digest * 31 + j.input();
        digest = digest * 31 + p;
        digest = digest * 31 + keys.hashCode();
        all.offered++;
        all.input += j.input();
        if (measured(t)) {
            tail.offered++;
            tail.input += j.input();
        }
        var stat = byPriority.computeIfAbsent(p, k -> new Stat());
        stat.offered++;
        stat.input += j.input();
    }

    void reject(Job j, long now) {
        all.rejected++;
        byPriority.get(j.priority).rejected++;
        if (measured(now)) tail.rejected++;
    }

    void finish(Worker w, long now) {
        for (Work x : w.active) {
            w.cache.admit(x.lease, x.job.keys);
            all.complete(x);
            byPriority.get(x.job.priority).complete(x);
            if (measured(now)) tail.complete(x);
            if (now + 1 > x.job.deadline) all.missedDeadline++;
        }
        w.active.clear();
        w.due = NEVER;
    }

    record Candidate(Job job, int hit, int fresh) {}

    static boolean sharesMissingPrefix(Candidate candidate, Work active) {
        int miss = Math.max(candidate.hit, active.hit) / 512;
        int limit = Math.min(candidate.job.keys.size(), active.job.keys.size());
        int common = 0;
        while (common < limit && candidate.job.keys.get(common).equals(active.job.keys.get(common)))
            common++;
        return common - miss >= 2 && common - miss > 2L * (limit - common);
    }

    void start(Worker w, long now) {
        if (w.due != NEVER || w.queue.isEmpty()) return;
        List<Candidate> choices = new ArrayList<>();
        int highest = Integer.MIN_VALUE;
        for (Job j : w.queue) {
            int hit = w.hit(j), fresh = j.input() - hit;
            if (now + price(fresh) + 1 > j.deadline) continue;
            if (j.priority > highest) {
                choices.clear();
                highest = j.priority;
            }
            if (j.priority == highest) choices.add(new Candidate(j, hit, fresh));
        }
        if (choices.isEmpty()) return;
        if (benefit)
            choices.sort(
                    (a, b) -> {
                        int cmp =
                                Long.compare(
                                        (long) a.fresh * b.job.input(),
                                        (long) b.fresh * a.job.input());
                        return cmp != 0 ? cmp : Integer.compare(a.job.id, b.job.id);
                    });
        int fresh = 0, input = 0;
        long earliest = NEVER;
        Set<Integer> branch = new HashSet<>();
        for (Candidate c : choices) {
            if (w.active.size() >= batchLimit) break;
            if (onePerBranch && branch.contains(c.job.session)) continue;
            if (overlap && w.active.stream().anyMatch(x -> sharesMissingPrefix(c, x))) continue;
            if (input + c.job.input() > 32 * 256 * 512) continue;
            if (now + price(fresh + c.fresh) + 1 > Math.min(earliest, c.job.deadline)) continue;
            var lease = w.cache.acquire(c.job.keys.size(), c.job.reusableKeys());
            if (lease == null) continue;
            // Other admissions can evict blocks between candidate scoring and reservation.
            int actualHit = lease.hitKeys.size() * 512, actualFresh = c.job.input() - actualHit;
            if (now + price(fresh + actualFresh) + 1 > Math.min(earliest, c.job.deadline)) {
                w.cache.release(lease);
                continue;
            }
            w.active.add(new Work(c.job, lease, actualHit, actualFresh));
            fresh += actualFresh;
            input += c.job.input();
            earliest = Math.min(earliest, c.job.deadline);
            w.queue.remove(c.job);
            waitingIds.remove(c.job.id);
            branch.add(c.job.session);
        }
        if (!w.active.isEmpty()) w.due = now + price(fresh);
    }

    int pending() {
        int n = 0;
        for (var w : workers) n += w.queue.size();
        return n;
    }

    void run() throws Exception {
        int target = Q * END / 1000;
        long sample = 1000;
        int q0 = -1, q1 = -1;
        while (true) {
            long t = next < target ? arrival(next) : NEVER;
            for (var w : workers) t = Math.min(t, w.due);
            if (sample <= stop() + DEADLINE + 1000) t = Math.min(t, sample);
            // All requests use the same relative deadline, so the arrival deque is time ordered.
            if (!deadlines.isEmpty()) t = Math.min(t, deadlines.getFirst().deadline);
            if (t == NEVER) break;
            while (!deadlines.isEmpty() && deadlines.getFirst().deadline == t) {
                var expired = deadlines.removeFirst();
                if (waitingIds.remove(expired.id)) {
                    workers[expired.session % N].queue.remove(expired);
                    reject(expired, t);
                }
            }
            for (var w : workers) if (w.due == t) finish(w, t);
            while (next < target && arrival(next) == t) enqueue((int) t);
            for (var w : workers) start(w, t);
            if (t == sample) {
                if (t == stop() / 2) q0 = pending();
                if (t == stop()) q1 = pending();
                sample += 1000;
            }
            if (next == target
                    && pending() == 0
                    && Arrays.stream(workers).allMatch(w -> w.due == NEVER)) break;
        }
        if (all.completed + all.rejected != target || all.missedDeadline != 0)
            throw new AssertionError("unsettled or late");
        int n = tail.lengths.size();
        Collections.sort(tail.lengths);
        double seconds = stop() / 2000.;
        System.out.printf(
                Locale.ROOT,
                "TAIL digest=%d sec=%.1f offered=%.2f complete=%.2f reject=%.2f offeredTokensM=%.3f"
                    + " goodTokensM=%.3f freshM=%.3f hit=%.2f avgLen=%.0f p50Len=%d p90Len=%d"
                    + " pending=%d->%d late=%d%n",
                digest,
                seconds,
                tail.offered / seconds,
                tail.completed / seconds,
                tail.rejected / seconds,
                tail.input / seconds / 1e6,
                tail.goodTokens / seconds / 1e6,
                tail.fresh / seconds / 1e6,
                100.0 * tail.cached / tail.goodTokens,
                (double) tail.goodTokens / n,
                tail.lengths.get(n / 2),
                tail.lengths.get(n * 9 / 10),
                q0,
                q1,
                all.missedDeadline);
        for (var kv : byPriority.entrySet()) {
            var x = kv.getValue();
            System.out.printf(
                    "PRIO %d offered=%d complete=%d reject=%d input=%d goodTokens=%d%n",
                    kv.getKey(), x.offered, x.completed, x.rejected, x.input, x.goodTokens);
        }
    }

    public static void main(String[] args) throws Exception {
        new ServiceControl(args[0], args[1], Integer.parseInt(args[2])).run();
    }
}
