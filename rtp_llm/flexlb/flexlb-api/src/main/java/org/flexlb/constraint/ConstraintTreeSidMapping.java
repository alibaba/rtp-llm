package org.flexlb.constraint;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Pattern;

/** A tokenizer-owned, immutable mapping. SID numbers are NOT token ids. */
public record ConstraintTreeSidMapping(
        @JsonProperty("mapping_fingerprint") String fingerprint,
        @JsonProperty("vocab_size") int vocabSize,
        @JsonProperty("start_token_id") int startTokenId,
        @JsonProperty("end_token_id") int endTokenId,
        Map<String, Integer> tokens) {

    private static final Pattern SYMBOL = Pattern.compile("C[0-9]+");

    public ConstraintTreeSidMapping validated() {
        if (tokens == null || tokens.isEmpty() || vocabSize <= 0 || startTokenId < 0 || startTokenId >= vocabSize
                || endTokenId < 0 || endTokenId >= vocabSize || startTokenId == endTokenId) {
            throw new IllegalArgumentException("Worker SID mapping is incomplete");
        }
        Map<String, Integer> sorted = new TreeMap<>(tokens);
        HashSet<Integer> ids = new HashSet<>();
        MessageDigest digest = newDigest();
        update(digest, "rtp-sid-mapping-v1\n" + vocabSize + "\n" + startTokenId + "\n" + endTokenId + "\n");
        for (var entry : sorted.entrySet()) {
            Integer id = entry.getValue();
            if (!SYMBOL.matcher(entry.getKey()).matches() || id == null || id < 0 || id >= vocabSize
                    || id == startTokenId || id == endTokenId || !ids.add(id)) {
                throw new IllegalArgumentException("Worker SID mapping contains invalid or aliased tokens");
            }
            update(digest, entry.getKey() + "\t" + id + "\n");
        }
        if (!HexFormat.of().formatHex(digest.digest()).equals(fingerprint)) {
            throw new IllegalArgumentException("Worker SID mapping fingerprint verification failed");
        }
        return new ConstraintTreeSidMapping(fingerprint, vocabSize, startTokenId, endTokenId, Map.copyOf(sorted));
    }

    public BuildRequest convert(BuildRequest request) {
        if ((request.startTokenId() != null && request.startTokenId() != startTokenId)
                || (request.endTokenId() != null && request.endTokenId() != endTokenId)) {
            throw new IllegalArgumentException("request start/end token ids disagree with Worker SID mapping");
        }
        if (!request.hasSids() || request.sids().get(0).matches("[0-9]+(?:"
                + Pattern.quote(request.resolvedSeparator()) + "[0-9]+)*")) {
            // Existing numeric inputs retain variable-length trie support. SARO uses the coded SID branch below.
            return new BuildRequest(request.version(), request.model(), startTokenId, endTokenId,
                    request.separator(), request.rqTokenIds(), request.sids());
        }
        var sequences = new ArrayList<int[]>(request.sids().size());
        for (int index = 0; index < request.sids().size(); index++) {
            String sid = request.sids().get(index);
            var matcher = SYMBOL.matcher(sid);
            int position = 0;
            var ids = new ArrayList<Integer>();
            while (matcher.find()) {
                if (matcher.start() != position) {
                    break;
                }
                Integer id = tokens.get(matcher.group());
                if (id == null) {
                    throw new IllegalArgumentException("unknown SID symbol at index " + index + ": " + matcher.group());
                }
                ids.add(id);
                position = matcher.end();
            }
            if (position != sid.length() || ids.size() != 2) {
                throw new IllegalArgumentException("SARO SID must contain exactly two C-token symbols at index " + index);
            }
            sequences.add(ids.stream().mapToInt(Integer::intValue).toArray());
        }
        return new BuildRequest(request.version(), request.model(), startTokenId, endTokenId, request.separator(),
                sequences, null);
    }

    static MessageDigest newDigest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    static void update(MessageDigest digest, String value) {
        digest.update(value.getBytes(StandardCharsets.UTF_8));
    }
}
