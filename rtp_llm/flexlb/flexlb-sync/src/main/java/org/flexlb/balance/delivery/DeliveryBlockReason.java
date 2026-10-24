package org.flexlb.balance.delivery;

import org.flexlb.balance.estimate.PrefillTtftEstimate;

import java.util.Objects;

/**
 * 一次投递准备没能继续的原因。
 * 投递等待可以附带额外的路由发布限制；
 * 没有 projectionSemantics 时，路由预估保持不变。
 */
public record DeliveryBlockReason(
        Status status,
        Availability availability,
        PrefillTtftEstimate.AdmissionBlockSemantics projectionSemantics,
        Throwable cause) {

    public static final DeliveryBlockReason OWNERSHIP_LOST =
            new DeliveryBlockReason(Status.OWNERSHIP_LOST, null, null, null);

    public DeliveryBlockReason {
        boolean valid = switch (status) {
            case UNAVAILABLE -> availability != null && cause == null;
            case FAILED -> cause != null && availability == null && projectionSemantics == null;
            case OWNERSHIP_LOST -> availability == null && cause == null && projectionSemantics == null;
        };
        if (!valid) {
            throw new IllegalArgumentException(
                    "delivery block reason status requires its exact payload");
        }
    }

    public static DeliveryBlockReason unavailable(
            Availability availability,
            PrefillTtftEstimate.AdmissionBlockSemantics projectionSemantics) {
        return new DeliveryBlockReason(Status.UNAVAILABLE,
                availability,
                Objects.requireNonNull(projectionSemantics, "projectionSemantics"),
                null);
    }

    /**
     * 暂停投递，但不限制请求继续发布到 worker 队列。
     */
    public static DeliveryBlockReason deliveryUnavailable(Availability availability) {
        return new DeliveryBlockReason(Status.UNAVAILABLE,
                availability, null, null);
    }

    public static DeliveryBlockReason failed(Throwable cause) {
        return new DeliveryBlockReason(Status.FAILED, null, null, cause);
    }

    public boolean unavailable() {
        return status == Status.UNAVAILABLE;
    }

    public enum Status {
        UNAVAILABLE,
        OWNERSHIP_LOST,
        FAILED
    }

    /**
     * 针对某一次不可用的准入，提供不阻塞的唤醒能力。
     * 实现在被查询时不能获取 endpoint 的修改锁。
     */
    public interface Availability {

        boolean isAvailable();

        void addListener(Runnable listener);

        void removeListener(Runnable listener);
    }

    /**
     * 要么是成功拿到的值（或能力），要么是一个被卡住的原因，二者有且只有一个。
     */
    public record Attempt<T>(T value, DeliveryBlockReason blockReason) {

        public Attempt {
            if ((value == null) == (blockReason == null)) {
                throw new IllegalArgumentException(
                        "attempt requires exactly one value or block reason");
            }
        }

        public static <T> Attempt<T> accepted(T value) {
            return new Attempt<>(value, null);
        }

        public static <T> Attempt<T> rejected(DeliveryBlockReason blockReason) {
            return new Attempt<>(null, blockReason);
        }

        public boolean accepted() {
            return value != null;
        }
    }
}
