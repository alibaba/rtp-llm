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
 * metadata and is never an ownership token.  A token-balanced Prefill
 * selection may also hold a soft workload reservation that the route owner
 * converts into pending token work at commit.</p>
 */
public final class SelectedRole implements AutoCloseable {

    private WorkerEndpoint.GenerationPin generationPin;
    private final ServerStatus serverStatus;
    private final long prefillWorkMs;
    private final long placementVersion;
    private PrefillTokenWorkload.Reservation tokenReservation;

    private SelectedRole(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs,
            long placementVersion,
            PrefillTokenWorkload.Reservation tokenReservation) {
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
        if (serverStatus.getRole() == RoleType.DECODE
                && !(endpoint instanceof DecodeEndpoint)) {
            throw new IllegalArgumentException(
                    "Decode selection requires a Decode endpoint role");
        }
        this.prefillWorkMs = prefillWorkMs;
        if (placementVersion < 0L) {
            throw new IllegalArgumentException(
                    "placementVersion must be non-negative");
        }
        this.placementVersion = placementVersion;
    }

    public static SelectedRole prefill(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs) {
        return prefill(
                generationPin, serverStatus, prefillWorkMs,
                endpointPlacementVersion(generationPin));
    }

    public static SelectedRole prefill(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs,
            long placementVersion) {
        return prefill(
                generationPin, serverStatus, prefillWorkMs,
                placementVersion, null);
    }

    /** Calling a factory consumes the pin, including every validation failure. */
    public static SelectedRole prefill(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs,
            long placementVersion,
            PrefillTokenWorkload.Reservation tokenReservation) {
        if (prefillWorkMs < 0L) {
            if (generationPin != null) {
                generationPin.close();
            }
            if (tokenReservation != null) {
                tokenReservation.close();
            }
            throw new IllegalArgumentException(
                    "Prefill work must be non-negative");
        }
        return createOwned(
                generationPin, serverStatus, prefillWorkMs,
                placementVersion, tokenReservation);
    }

    public static SelectedRole decode(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus) {
        return decode(
                generationPin, serverStatus,
                endpointPlacementVersion(generationPin));
    }

    public static SelectedRole decode(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long placementVersion) {
        if (serverStatus == null || serverStatus.getRole() != RoleType.DECODE) {
            if (generationPin != null) {
                generationPin.close();
            }
            throw new IllegalArgumentException("Decode selection requires Decode metadata");
        }
        return createOwned(generationPin, serverStatus, -1L, placementVersion, null);
    }

    public static SelectedRole stateless(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus) {
        return createOwned(
                generationPin, serverStatus, -1L, 0L, null);
    }

    /** Calling a factory consumes the pin, including every validation failure. */
    private static SelectedRole createOwned(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs,
            long placementVersion) {
        return createOwned(generationPin, serverStatus, prefillWorkMs,
                placementVersion, null);
    }

    private static SelectedRole createOwned(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs,
            long placementVersion,
            PrefillTokenWorkload.Reservation tokenReservation) {
        try {
            return new SelectedRole(
                    generationPin, serverStatus, prefillWorkMs,
                    placementVersion, tokenReservation);
        } catch (RuntimeException | Error failure) {
            if (generationPin != null) {
                generationPin.close();
            }
            if (tokenReservation != null) {
                tokenReservation.close();
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

    public long placementVersion() {
        return placementVersion;
    }

    private static long endpointPlacementVersion(
            WorkerEndpoint.GenerationPin pin) {
        if (pin == null) {
            return 0L;
        }
        WorkerEndpoint endpoint = pin.endpoint();
        if (endpoint instanceof PrefillEndpoint prefill) {
            return prefill.placementVersion();
        }
        if (endpoint instanceof DecodeEndpoint decode) {
            return decode.placementVersion();
        }
        return 0L;
    }

    /** Move the exact pin to the next domain owner. */
    public WorkerEndpoint.GenerationPin takeGenerationPin() {
        WorkerEndpoint.GenerationPin owned = requireOwnedPin();
        generationPin = null;
        return owned;
    }

    /** Move the soft token reservation to the route owner; never consumes admission capacity. */
    public PrefillTokenWorkload.Reservation takeTokenReservation() {
        PrefillTokenWorkload.Reservation owned = tokenReservation;
        tokenReservation = null;
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
        closeTokenReservation();
    }

    private void closeTokenReservation() {
        PrefillTokenWorkload.Reservation reservation = tokenReservation;
        tokenReservation = null;
        if (reservation != null) {
            reservation.close();
        }
    }
}
