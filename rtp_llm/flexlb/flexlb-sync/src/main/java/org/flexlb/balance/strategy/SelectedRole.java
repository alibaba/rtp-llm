package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.Objects;

/**
 * One exact endpoint-generation selection.
 *
 * <p>The selection owns its generation pin until the router moves it into a
 * DIRECT registration or a queue-route admission.  It carries only immutable
 * routing output besides that pin; {@link ServerStatus} remains response
 * metadata and is never an ownership token.</p>
 */
public final class SelectedRole implements AutoCloseable {

    private WorkerEndpoint.GenerationPin generationPin;
    private final ServerStatus serverStatus;
    private final long prefillWorkMs;
    private final long decodeTotalKv;

    private SelectedRole(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs,
            long decodeTotalKv) {
        this.generationPin = generationPin;
        this.serverStatus = serverStatus;
        if (!serverStatus.isSuccess()) {
            throw new IllegalArgumentException(
                    "SelectedRole requires successful response metadata");
        }
        WorkerEndpoint endpoint = generationPin.endpoint();
        if (endpoint.getStatus().getGenerationId()
                != generationPin.generationId()) {
            throw new IllegalArgumentException(
                    "selection pin does not match endpoint generation");
        }
        if (!Objects.equals(serverStatus.getServerIp(), endpoint.getIp())
                || serverStatus.getHttpPort() != endpoint.getHttpPort()) {
            throw new IllegalArgumentException(
                    "selection metadata does not match pinned endpoint address");
        }
        if (prefillWorkMs >= 0L
                && (!(endpoint instanceof PrefillEndpoint)
                        || serverStatus.getRole() != RoleType.PREFILL
                                && serverStatus.getRole() != RoleType.PDFUSION)) {
            throw new IllegalArgumentException(
                    "Prefill selection requires a Prefill endpoint role");
        }
        if (decodeTotalKv >= 0L
                && (!(endpoint instanceof DecodeEndpoint)
                        || serverStatus.getRole() != RoleType.DECODE)) {
            throw new IllegalArgumentException(
                    "Decode selection requires a Decode endpoint role");
        }
        this.prefillWorkMs = prefillWorkMs;
        this.decodeTotalKv = decodeTotalKv;
    }

    public static SelectedRole prefill(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs) {
        if (prefillWorkMs < 0L) {
            if (generationPin != null) {
                generationPin.close();
            }
            throw new IllegalArgumentException(
                    "Prefill work must be non-negative");
        }
        return createOwned(
                generationPin, serverStatus, prefillWorkMs, -1L);
    }

    public static SelectedRole decode(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long decodeTotalKv) {
        return createOwned(
                generationPin, serverStatus, -1L,
                Math.max(0L, decodeTotalKv));
    }

    public static SelectedRole stateless(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus) {
        return createOwned(
                generationPin, serverStatus, -1L, -1L);
    }

    /** Calling a factory consumes the pin, including every validation failure. */
    private static SelectedRole createOwned(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs,
            long decodeTotalKv) {
        try {
            return new SelectedRole(
                    generationPin, serverStatus, prefillWorkMs, decodeTotalKv);
        } catch (RuntimeException | Error failure) {
            if (generationPin != null) {
                generationPin.close();
            }
            throw failure;
        }
    }

    public ServerStatus serverStatus() {
        return serverStatus;
    }

    public long prefillWorkMs() {
        if (prefillWorkMs < 0L) {
            throw new IllegalStateException(
                    "selection does not carry Prefill work");
        }
        return prefillWorkMs;
    }

    public long decodeTotalKv() {
        if (decodeTotalKv < 0L) {
            throw new IllegalStateException(
                    "selection does not carry Decode capacity");
        }
        return decodeTotalKv;
    }

    /** Move the exact pin to the next domain owner. */
    public WorkerEndpoint.GenerationPin takeGenerationPin() {
        WorkerEndpoint.GenerationPin owned = requireOwnedPin();
        generationPin = null;
        return owned;
    }

    private WorkerEndpoint.GenerationPin requireOwnedPin() {
        if (generationPin == null) {
            throw new IllegalStateException(
                    "selected endpoint generation was already consumed");
        }
        return generationPin;
    }

    @Override
    public void close() {
        WorkerEndpoint.GenerationPin owned = generationPin;
        generationPin = null;
        if (owned != null) {
            owned.close();
        }
    }
}
