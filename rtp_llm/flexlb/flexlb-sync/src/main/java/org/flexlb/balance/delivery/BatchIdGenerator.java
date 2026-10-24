package org.flexlb.balance.delivery;

import org.flexlb.util.Logger;

import java.lang.management.ManagementFactory;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongSupplier;

/**
 * 生成 batch id，格式与 Snowflake 兼容：31 位时间戳（秒）、12 位节点号、21 位序号。
 * 节点号由本机 IP、端口、进程号和进程启动时间的哈希得到，用来避免多个 FlexLB 实例生成相同的 id。
 */
public final class BatchIdGenerator implements LongSupplier {

    private static final int TIMESTAMP_BITS = 31;
    private static final int NODE_ID_BITS = 12;
    private static final int SEQUENCE_BITS = 21;
    private static final int NODE_ID_SHIFT = SEQUENCE_BITS;
    private static final int TIMESTAMP_SHIFT = NODE_ID_BITS + SEQUENCE_BITS;
    private static final long SEQUENCE_MASK = (1L << SEQUENCE_BITS) - 1L;
    private static final long EPOCH_SECONDS = 1_577_836_800L;
    private static final String LOOPBACK_ADDRESS = "127.0.0.1";

    private final long nodeId;
    private final AtomicLong sequence = new AtomicLong();

    BatchIdGenerator(String localIp, int port, long processId, long processStartTimeMs) {
        this.nodeId = computeNodeId(localIp, port, processId, processStartTimeMs);
    }

    public static BatchIdGenerator create(int port) {
        return new BatchIdGenerator(detectLocalIp(),
                                    port,
                                    ProcessHandle.current().pid(),
                                    ManagementFactory.getRuntimeMXBean().getStartTime());
    }

    @Override
    public long getAsLong() {
        long timestamp = TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis()) - EPOCH_SECONDS;
        if (timestamp < 0L) {
            throw new IllegalStateException("system clock is before the batch ID epoch");
        }
        if ((timestamp >>> TIMESTAMP_BITS) != 0L) {
            throw new IllegalStateException("batch ID timestamp overflow");
        }
        return (timestamp << TIMESTAMP_SHIFT)
                | (nodeId << NODE_ID_SHIFT)
                | (sequence.getAndIncrement() & SEQUENCE_MASK);
    }

    /**
     * 取 SHA-256 结果的最后两个字节，其中高字节只保留低 4 位，凑成 12 位节点号。
     */
    private static long computeNodeId(String localIp, int port, long processId, long processStartTimeMs) {
        String input = localIp + ":" + port + ":" + processId + ":" + processStartTimeMs;
        byte[] hash;
        try {
            hash = MessageDigest.getInstance("SHA-256").digest(input.getBytes(StandardCharsets.UTF_8));
        } catch (NoSuchAlgorithmException failure) {
            throw new IllegalStateException("SHA-256 not available", failure);
        }
        int high = hash[hash.length - 2] & 0x0F;
        int low = hash[hash.length - 1] & 0xFF;
        return ((long) high << Byte.SIZE) | low;
    }

    private static String detectLocalIp() {
        try {
            return InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException failure) {
            Logger.warn("Failed to detect local IP for batch ids; using loopback", failure);
            return LOOPBACK_ADDRESS;
        }
    }
}
