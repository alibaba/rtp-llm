package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Failures;

import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Frozen Encoder metadata and the generation pin held until admission or route handoff.
 */
public final class EncoderSelection implements AutoCloseable {
    private WorkerEndpoint.GenerationPin pin;
    private final ServerStatus status;

    private EncoderSelection(WorkerEndpoint.GenerationPin pin, ServerStatus status) {
        checkState(pin != null && pin.isOpen(), "Encoder generation is closed");
        this.status = ServerStatus.copyOf(Objects.requireNonNull(status, "Encoder metadata"));
        checkArgument(
                this.status.isSuccess()
                        && this.status.getRole() == RoleType.ENCODER
                        && pin.endpoint() instanceof EncoderEndpoint,
                "Encoder selection requires successful Encoder metadata");
        checkArgument(
                Objects.equals(this.status.getServerIp(), pin.endpoint().getIp())
                        && this.status.getHttpPort() == pin.endpoint().getHttpPort(),
                "Encoder metadata does not match pinned endpoint");
        this.pin = pin;
    }

    /**
     * Consumes the generation pin, including validation failures.
     */
    public static EncoderSelection create(WorkerEndpoint.GenerationPin pin, ServerStatus status) {
        try {
            return new EncoderSelection(pin, status);
        } catch (RuntimeException | Error failure) {
            Failures.append(failure, Failures.close(pin));
            throw failure;
        }
    }

    public ServerStatus serverStatus() {
        return ServerStatus.copyOf(status);
    }

    public WorkerEndpoint.GenerationPin generationPin() {
        checkState(pin != null && pin.isOpen(), "Encoder selection no longer holds an open pin");
        return pin;
    }

    /**
     * Transfers the pin to a combined route; closing this selection then leaves it with that route.
     */
    public ServerStatus transferTo(RequestRoute route) {
        var selected = generationPin();
        pin = null;
        return route.add(selected, status, -1L, 0L);
    }

    @Override
    public void close() {
        if (pin != null) {
            pin.close();
        }
    }
}
