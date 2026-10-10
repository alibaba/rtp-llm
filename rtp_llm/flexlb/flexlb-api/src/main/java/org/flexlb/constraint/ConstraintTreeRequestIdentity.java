package org.flexlb.constraint;

import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;

import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;

/** Canonical set identity: SID order/duplicates do not change a submission. */
final class ConstraintTreeRequestIdentity {
    private ConstraintTreeRequestIdentity() { }

    static BuildRequest freeze(BuildRequest request) {
        if (request.sids() != null && request.sids().stream().anyMatch(java.util.Objects::isNull)) {
            throw new IllegalArgumentException("sids contains null");
        }
        List<int[]> sequences = null;
        if (request.rqTokenIds() != null) {
            sequences = new ArrayList<>(request.rqTokenIds().size());
            for (int[] sequence : request.rqTokenIds()) {
                if (sequence == null) {
                    throw new IllegalArgumentException("rq_token_ids contains null");
                }
                sequences.add(sequence.clone());
            }
            sequences = List.copyOf(sequences);
        }
        return new BuildRequest(request.version(), request.model(), request.startTokenId(), request.endTokenId(),
                request.separator(), sequences, request.sids() == null ? null : List.copyOf(request.sids()));
    }

    static String fingerprint(BuildRequest request) {
        MessageDigest digest = ConstraintTreeSidMapping.newDigest();
        add(digest, "rtp-sid-input-v1");
        add(digest, request.model());
        add(digest, Integer.toString(request.resolvedStartTokenId()));
        add(digest, Integer.toString(request.resolvedEndTokenId()));
        add(digest, request.resolvedSeparator());
        add(digest, request.hasSids() ? "sids" : "rq_token_ids");
        String[] values = request.hasSids() ? request.sids().toArray(String[]::new)
                : request.rqTokenIds().stream().map(Arrays::toString).toArray(String[]::new);
        Arrays.sort(values);
        String previous = null;
        for (String value : values) {
            if (!value.equals(previous)) {
                add(digest, value);
                previous = value;
            }
        }
        return HexFormat.of().formatHex(digest.digest());
    }

    private static void add(MessageDigest digest, String value) {
        // Length-delimited UTF-8 avoids ambiguous separators inside untrusted strings.
        byte[] bytes = value.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        ConstraintTreeSidMapping.update(digest, bytes.length + ":");
        digest.update(bytes);
    }
}
