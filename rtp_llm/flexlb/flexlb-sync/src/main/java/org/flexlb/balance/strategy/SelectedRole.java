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
 * metadata and is never an ownership token. A Prefill selection transfers
 * its planned work and optional hotspot migration charge to the route owner.</p>
 */
public final class SelectedRole implements AutoCloseable {

    private WorkerEndpoint.GenerationPin generationPin;
    private final ServerStatus serverStatus;
    private final long prefillWorkMs;
    private final long placementVersion;
    private final long residentPrefixTokens;
    private PrefillTokenWorkload.Reservation tokenReservation;
    private HotspotRedirect hotspotRedirect;

    private SelectedRole(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs,
            long placementVersion, long residentPrefixTokens,
            PrefillTokenWorkload.Reservation tokenReservation, HotspotRedirect hotspotRedirect) {
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
        this.residentPrefixTokens = residentPrefixTokens;
        this.tokenReservation = tokenReservation;
        this.hotspotRedirect = hotspotRedirect;
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
        return prefill(generationPin, serverStatus, prefillWorkMs, placementVersion, 0L, tokenReservation, null);
    }

    static SelectedRole prefill(WorkerEndpoint.GenerationPin generationPin, ServerStatus serverStatus,
            long prefillWorkMs, long placementVersion, long residentPrefixTokens,
            PrefillTokenWorkload.Reservation tokenReservation, HotspotRedirect hotspotRedirect) {
        if (prefillWorkMs < 0L) {
            try (generationPin; tokenReservation; hotspotRedirect) {
                throw new IllegalArgumentException("Prefill work must be non-negative");
            }
        }
        return createOwned(generationPin, serverStatus, prefillWorkMs,
                placementVersion, residentPrefixTokens, tokenReservation, hotspotRedirect);
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
        return createOwned(generationPin, serverStatus, -1L, placementVersion, 0L, null, null);
    }

    public static SelectedRole stateless(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus) {
        return createOwned(
                generationPin, serverStatus, -1L, 0L, 0L, null, null);
    }

    private static SelectedRole createOwned(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs,
            long placementVersion, long residentPrefixTokens,
            PrefillTokenWorkload.Reservation tokenReservation, HotspotRedirect hotspotRedirect) {
        try {
            return new SelectedRole(
                    generationPin, serverStatus, prefillWorkMs,
                    placementVersion, residentPrefixTokens, tokenReservation, hotspotRedirect);
        } catch (RuntimeException | Error failure) {
            try (generationPin; tokenReservation; hotspotRedirect) {
                throw failure;
            }
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

    /** Resident KV match before accounting for work on the last prompt block. */
    public long residentPrefixTokens() { return residentPrefixTokens; }

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

    /** Transfer the optional hotspot migration charge with the Prefill selection. */
    public HotspotRedirect takeHotspotRedirect() {
        var owned = hotspotRedirect;
        hotspotRedirect = null;
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
        try (owned; var tokens = takeTokenReservation(); var redirect = takeHotspotRedirect()) { }
    }
}
