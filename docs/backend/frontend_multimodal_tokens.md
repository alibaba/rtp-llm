# Frontend-expanded multimodal tokens

In a frontend + separated ViT deployment, the frontend already obtains row
hashes and segment lengths from its selected ViT before routing prefill. It
now retains the compact expanded token sequence and sends it in
`GenerateInputPB.token_ids`, together with `multimodal_token_layout`:

```text
spans: [{offset: 2, length: 256}, {offset: 260, length: 128}]
```

Each span describes one embedding segment in the expanded sequence. Ordering
matches the flattened ViT output, including every segment/frame of a video
and every occurrence of repeated media. Offsets are not positions in the
original placeholder sequence. Media URLs and preprocessing options are still
sent so prefill can request the corresponding embeddings from ViT.

Prefill fetches embeddings, position IDs and extra inputs through the existing
ViT interface. With a token layout present, it validates segment counts,
lengths, bounds, ordering and the returned row hashes, then constructs the
text mask and multimodal locations directly. It does not rescan separator
tokens or splice another expanded sequence. Model-specific position IDs still
come from ViT; insertion offsets are not a substitute for them. Invalid
positions or lengths return `MM_WRONG_FORMAT_ERROR` before publishing features
or entering model execution. If an evicted embedding was recomputed with the
same shape but different hashes, prefill refreshes those token ranges from
the actual ViT hashes. Routing may have used an older cache hint, but the
prefill/decode KV-cache keys describe the fetched data. Matching hashes reuse
the incoming token storage; changed hashes copy the token buffer only after
validating all segments.

The frontend uses the same expanded values for cache routing and model RPC.
Original prompt tokens remain available for rerouting and response handling.
On ViT reselection, both expanded tokens and spans are replaced; a failed
route cannot reuse a previous expansion. Large metadata objects are released
before scheduling. The retained expanded array uses four bytes per token,
plus the small span list, rather than a list of boxed Python integers.

Requests without a layout retain the existing backend expansion path. Update
prefill/PDFUSION servers before enabling the new frontend version: an older
backend does not understand already-expanded multimodal requests. Rebuild the
Python and C++ protobuf outputs from the same updated `.proto` file.

This change does not alter `PrefillRpcServer` allocation retries or cache
embedding reuse across those retries.

Focused tests cover frontend serialization, multiple video segments, repeated
media, route replacement/failure, protobuf conversion, equality with legacy
prefill assembly, separator-like hash values, changed hashes after embedding
eviction, and rejection of malformed hashes or layouts.
