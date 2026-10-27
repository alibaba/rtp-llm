"""D3 video-group consumer tests: independence of batching, caching and order.

The d3 contract sends one engine multimodal input per 2-frame temporal group
(envelope URL + pre-resized uint8 frame tensor). These tests pin the properties
that make that safe, each against an independent reference implementation:

1. A group's EMBEDDINGS are identical whether it is embedded alone, in any
   batch, in any order, or duplicated — the Qwen3.5 vision tower segments
   attention per frame group (cu_seqlens = repeat_interleave(h*w, t)) and its
   rotary/absolute positions are spatial-only, repeated per frame. Verified
   against the REAL Qwen3.5 MoE vision tower on CPU, including against the
   canonical whole-clip grid [G, h, w] sliced per group.

2. A group's language-model MRoPE positions are LOCAL (t=0, h, w) — the engine
   adds each timestamp-separated span's base offset when it splices positions
   into the sequence. Verified against a port of the OFFICIAL transformers
   5.17.0 Qwen3.5 get_rope_index (modeling_qwen3_5.py:1378; get_rope_index
   splits video_grid_thw into one [1, h, w] row per timestamp-separated group,
   get_vision_position_ids:1326 gives each span t=start_position+h/w offsets
   and current_pos advances by max(h, w) // merge per span), composed with a
   port of the engine's C++ gatherer (PositionIdsGenerator::generatePositionIds,
   MROPE branch, rtp_llm/cpp/models/position_ids/PositionIdsGenerator.cc).
   Group metadata (group_index/total_groups) is therefore NOT part of the
   position contract — a media-cache hit replaying one group cannot change any
   position.
"""

import itertools
import pickle
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch

from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeImageEmbedding,
    d3_video_group_meta,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
    Qwen3_5MoeVisionConfig,
)
from rtp_llm.multimodal.multimodal_util import (
    encode_d3_video_group_url,
    is_d3_video_group_url,
    parse_d3_video_group_url,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType, VitParameters

# 64x64 frames with patch 16 / merge 2: a 4x4 patch grid, one 2x2 merged token
# row per frame group -> 4 tokens per group.
FRAME = 64
FRAME_W = 64
PATCH = 16
MERGE = 2
GROUP_TOKENS = (FRAME // (PATCH * MERGE)) * (FRAME_W // (PATCH * MERGE))

TEXT, IMAGE, VIDEO = 0, 1, 2


def make_meta(group_index, total_groups, fps=24.0, height=FRAME, width=FRAME_W):
    return {
        "version": 3,
        "mode": "d3_video_group",
        "video_key": "clip",
        "group_index": group_index,
        "total_groups": total_groups,
        "fps": fps,
        "frames_indices": [2 * group_index, 2 * group_index + 1],
        "height": height,
        "width": width,
        "tokens": (height // (PATCH * MERGE)) * (width // (PATCH * MERGE)),
    }


def group_frames(seed=0, height=FRAME, width=FRAME_W):
    """Deterministic uint8 [2, 3, height, width] frames for one group."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 256, (2, 3, height, width), dtype=torch.uint8, generator=generator
    )


# ---------------------------------------------------------------------------------------------
# Reference implementations (ported with provenance; the tests must NOT compare
# the code under test with itself)
# ---------------------------------------------------------------------------------------------


def official_get_vision_position_ids(start_position, grid_thw, spatial_merge_size):
    """transformers 5.17.0, qwen3_5/modeling_qwen3_5.py:1326-1376.

    ``position_temporal = arange(t) * time_interval`` (time_interval=1),
    ``position_height = arange(h) + start``, ``position_width = arange(w) + start``,
    meshgrid, then ``vision_position_ids[0] += start_position``.
    """
    t, h, w = (int(v) for v in grid_thw)
    gh, gw = h // spatial_merge_size, w // spatial_merge_size
    pos_t = torch.arange(t)
    pos_h = torch.arange(gh) + start_position
    pos_w = torch.arange(gw) + start_position
    t_grid, h_grid, w_grid = torch.meshgrid(pos_t, pos_h, pos_w, indexing="ij")
    vision = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1)
    vision[0] += start_position
    return vision


def official_get_rope_index(
    token_types, image_grids, video_grids, spatial_merge_size=2
):
    """transformers 5.17.0, qwen3_5/modeling_qwen3_5.py:1378-1480.

    Splits each video grid row [t, h, w] into t rows [1, h, w] (timestamps
    separate the groups), walks consecutive same-type token spans, gives text
    spans arange(len)+current_pos on all three components and media spans
    get_vision_position_ids(current_pos, grid), advancing current_pos by
    max(h, w) // spatial_merge_size per media span.
    """
    video_rows = []
    for grid in video_grids:
        video_rows.extend([[1, grid[1], grid[2]] for _ in range(grid[0])])
    iters = {IMAGE: iter(image_grids), VIDEO: iter(video_rows)}
    spans = []
    for key, group in itertools.groupby(enumerate(token_types), lambda item: item[1]):
        group = list(group)
        spans.append((key, group[0][0], group[-1][0] + 1))
    current_pos = 0
    rows = []
    for modality, start, end in spans:
        if modality == TEXT:
            length = end - start
            base = torch.arange(length) + current_pos
            rows.append(torch.stack([base, base, base]))
            current_pos += length
        else:
            grid = next(iters[modality])
            rows.append(
                official_get_vision_position_ids(current_pos, grid, spatial_merge_size)
            )
            current_pos += max(grid[1], grid[2]) // spatial_merge_size
    return torch.cat(rows, dim=1)


def engine_generate_position_ids(token_types, mm_spans):
    """rtp_llm/cpp/models/position_ids/PositionIdsGenerator.cc, MROPE branch.

    ``mm_spans`` is a list of ``(start_index, local_positions[n, 3])`` in
    sequence order. Text tokens take all three components from the running id;
    a span takes local + running id, then the running id becomes the max of the
    span's last position components + 1.
    """
    total = len(token_types)
    res = torch.empty(total, 3, dtype=torch.int32)
    now_pos = now_id = mm_idx = 0
    locs = [span[0] for span in mm_spans]
    while now_pos < total:
        if mm_idx >= len(locs) or now_pos < locs[mm_idx]:
            res[now_pos, 0] = res[now_pos, 1] = res[now_pos, 2] = now_id
            now_pos += 1
            now_id += 1
        else:
            local = mm_spans[mm_idx][1]
            feature_len = local.shape[0]
            res[now_pos : now_pos + feature_len] = (local + now_id).to(torch.int32)
            now_pos += feature_len
            now_id = int(res[now_pos - 1].max()) + 1
            mm_idx += 1
    return res


def d3_sequence(image_grid=None, video_grids=(), text_gaps=(7, 5, 9)):
    """A d3-style token-type sequence: text, media spans separated by text.

    ``video_grids`` holds one [t, h, w] patch grid per video; every group of
    every video becomes its own timestamp-separated VIDEO span (the official
    get_rope_index splits video grids exactly this way). ``text_gaps`` is
    (leading, between, trailing) text length. Returns (token_types, mm_spans).
    """
    types = [TEXT] * text_gaps[0]
    spans = []
    index = text_gaps[0]
    for grid in video_grids:
        for _ in range(grid[0]):
            local = local_positions([1, grid[1], grid[2]])
            spans.append((index, local))
            types.extend([VIDEO] * local.shape[0])
            index += local.shape[0]
            types.extend([TEXT] * text_gaps[1])
            index += text_gaps[1]
    if image_grid is not None:
        local = local_positions(image_grid)
        spans.append((index, local))
        types.extend([IMAGE] * local.shape[0])
        index += local.shape[0]
    types.extend([TEXT] * text_gaps[2])
    return types, spans


def local_positions(grid_thw, spatial_merge_size=MERGE):
    """The mixin's per-span local positions: t=0, h, w (get_position_ids of a
    [1, h, w] row) — what embedding()/batched_embedding return per group."""
    t, h, w = (int(v) for v in grid_thw)
    gh, gw = h // spatial_merge_size, w // spatial_merge_size
    t_grid = torch.zeros(gh * gw, dtype=torch.int32)
    h_grid = torch.arange(gh, dtype=torch.int32).view(gh, 1).expand(gh, gw).reshape(-1)
    w_grid = torch.arange(gw, dtype=torch.int32).view(1, gw).expand(gh, gw).reshape(-1)
    return torch.stack([t_grid, h_grid, w_grid], dim=1)


class D3VideoGroupEnvelopeTest(TestCase):
    def _encode(self, frames, group_index=3, total_groups=5, seed_meta=None):
        return encode_d3_video_group_url(
            video_key="clip",
            group_index=group_index,
            total_groups=total_groups,
            fps=24.0,
            frames_indices=[2 * group_index, 2 * group_index + 1],
            height=FRAME,
            width=FRAME_W,
            tokens=GROUP_TOKENS,
            frames=frames,
        )

    def test_roundtrip_returns_the_exact_frame_bytes(self):
        raw = bytes(group_frames(seed=11).numpy().tobytes())
        url = self._encode(raw)
        self.assertTrue(is_d3_video_group_url(url))
        parsed = parse_d3_video_group_url(url)
        meta = make_meta(3, 5)
        for key, value in meta.items():
            self.assertEqual(parsed[key], value, key)
        # The frames cross the envelope byte-exactly (lossless zlib payload):
        # same bytes, already producer-resized, no second decode/resize.
        self.assertEqual(parsed["frames"], raw)
        self.assertEqual(parsed["frames_bytes"], len(raw))
        # Deterministic: the same clip encodes to the same URL (engine media
        # cache keys are URL-derived).
        self.assertEqual(url, self._encode(raw))

    def test_rejects_broken_envelopes(self):
        import base64
        import json

        raw = bytes(group_frames(seed=12).numpy().tobytes())
        url = self._encode(raw)
        self.assertFalse(is_d3_video_group_url("data:video/mp4;base64,AAAA"))
        self.assertFalse(is_d3_video_group_url("frames-pack:base64,AAAA"))

        # Tampering happens on the RAW json envelope (frames still the base64
        # str, as on the wire).
        def raw_envelope(url):
            import base64 as b64mod
            import json as jsonmod

            return jsonmod.loads(
                b64mod.b64decode(url[len("frames-pack:base64,") :]).decode("utf-8")
            )

        base_env = raw_envelope(url)
        cases = {
            "version": 2,  # the pre-transport metadata-only envelope
            "mode": "other",
            "video_key": "",
            "group_index": 5,  # == total_groups: out of range (0 <= i < total)
            "total_groups": 0,
            "frames_indices": [0],
            "height": 63,
            "width": 0,
            "tokens": 5,
            "fps": 0,
            "frames_bytes": 2 * 3 * FRAME * FRAME_W + 1,
            "frames": "not base64!!",
        }
        for key, value in cases.items():
            bad = dict(base_env)
            bad[key] = value
            bad_url = (
                "frames-pack:base64,"
                + base64.b64encode(json.dumps(bad).encode()).decode()
            )
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    parse_d3_video_group_url(bad_url)
        for key in ("frames", "frames_bytes"):
            bad = dict(base_env)
            del bad[key]
            bad_url = (
                "frames-pack:base64,"
                + base64.b64encode(json.dumps(bad).encode()).decode()
            )
            with self.subTest(missing=key):
                with self.assertRaises(ValueError):
                    parse_d3_video_group_url(bad_url)
        # Valid indices MUST parse, including middle ones: partial media-cache
        # hits, reordering and duplication all rely on every
        # 0 <= group_index < total_groups working (nothing stricter — the old
        # fixture's group_index=4 of 5 was valid, which is why it never
        # raised).
        for good in (0, 2, 4):
            ok = dict(base_env)
            ok["group_index"] = good
            ok_url = (
                "frames-pack:base64,"
                + base64.b64encode(json.dumps(ok).encode()).decode()
            )
            with self.subTest(good_index=good):
                self.assertEqual(parse_d3_video_group_url(ok_url)["group_index"], good)

    def test_encode_rejects_frames_that_do_not_match_the_declared_size(self):
        with self.assertRaises(ValueError):
            encode_d3_video_group_url(
                "clip", 0, 4, 24.0, [0, 1], FRAME, FRAME_W, GROUP_TOKENS, b"short"
            )

    def test_oversized_advertised_dims_are_rejected_before_decoding(self):
        """The official per-frame pixel ceiling is enforced on both sides.

        The producer never emits frames beyond 200,704 pixels per frame
        (video_resize_dims' cap_pixels_per_frame port), so a forged envelope
        advertising larger dims must be rejected at the DIMS bound — before
        the frames payload is sized, decoded or inflated.
        """
        import base64
        import json
        import zlib

        # 224x928 = 207,872 pixels > the 200,704 ceiling; %32-aligned so only
        # the pixel bound fires.
        big_h, big_w = 224, 928
        self.assertGreater(big_h * big_w, 200_704)
        with self.assertRaises(ValueError):
            encode_d3_video_group_url(
                "clip",
                0,
                4,
                24.0,
                [0, 1],
                big_h,
                big_w,
                (big_h // 32) * (big_w // 32),
                bytes(2 * 3 * big_h * big_w),
            )
        envelope = {
            "version": 3,
            "mode": "d3_video_group",
            "video_key": "clip",
            "group_index": 0,
            "total_groups": 4,
            "fps": 24.0,
            "frames_indices": [0, 1],
            "height": big_h,
            "width": big_w,
            "tokens": (big_h // 32) * (big_w // 32),
            "frames_bytes": 2 * 3 * big_h * big_w,
            # A payload that would decompress to the advertised size; it must
            # never be reached.
            "frames": base64.b64encode(
                zlib.compress(bytes(2 * 3 * big_h * big_w))
            ).decode(),
        }
        bad_url = (
            "frames-pack:base64,"
            + base64.b64encode(json.dumps(envelope).encode()).decode()
        )
        with self.assertRaises(ValueError) as caught:
            parse_d3_video_group_url(bad_url)
        self.assertIn("per-frame ceiling", str(caught.exception))

    def test_frames_payload_expanding_beyond_the_declared_size_is_rejected(self):
        """Bounded inflation: a stream that expands past the declared frames
        is rejected by the decompressor's output cap, not by a length check
        after allocating."""
        import base64
        import json
        import zlib

        raw = bytes(group_frames(seed=41).numpy().tobytes())
        url = encode_d3_video_group_url(
            "clip", 0, 4, 24.0, [0, 1], FRAME, FRAME_W, GROUP_TOKENS, frames=raw
        )
        envelope = json.loads(
            base64.b64decode(url[len("frames-pack:base64,") :]).decode("utf-8")
        )
        # The payload inflates to twice the declared frames.
        envelope["frames"] = base64.b64encode(zlib.compress(raw + raw)).decode()
        bomb = (
            "frames-pack:base64,"
            + base64.b64encode(json.dumps(envelope).encode()).decode()
        )
        with self.assertRaises(ValueError) as caught:
            parse_d3_video_group_url(bomb)
        self.assertIn("expands beyond", str(caught.exception))

    def test_truncated_and_trailing_payload_streams_are_rejected(self):
        import base64
        import json
        import zlib

        raw = bytes(group_frames(seed=42).numpy().tobytes())
        url = encode_d3_video_group_url(
            "clip", 0, 4, 24.0, [0, 1], FRAME, FRAME_W, GROUP_TOKENS, frames=raw
        )
        envelope = json.loads(
            base64.b64decode(url[len("frames-pack:base64,") :]).decode("utf-8")
        )
        compressed = zlib.compress(raw)
        cases = {
            "truncated": base64.b64encode(compressed[: len(compressed) // 2]).decode(),
            "trailing": base64.b64encode(compressed + b"junk").decode(),
        }
        for name, payload in cases.items():
            with self.subTest(case=name):
                envelope["frames"] = payload
                bad_url = (
                    "frames-pack:base64,"
                    + base64.b64encode(json.dumps(envelope).encode()).decode()
                )
                with self.assertRaises(ValueError) as caught:
                    parse_d3_video_group_url(bad_url)
                self.assertIn("truncated or carries trailing", str(caught.exception))

    def test_oversized_encoded_payload_is_rejected_before_decoding(self):
        """The ENCODED payload is bounded before any base64 decode: a
        compressed stream never usefully exceeds the raw frame size, so a
        longer one is rejected without allocating."""
        import base64
        import json

        raw = bytes(group_frames(seed=43).numpy().tobytes())
        url = encode_d3_video_group_url(
            "clip", 0, 4, 24.0, [0, 1], FRAME, FRAME_W, GROUP_TOKENS, frames=raw
        )
        envelope = json.loads(
            base64.b64decode(url[len("frames-pack:base64,") :]).decode("utf-8")
        )
        envelope["frames"] = envelope["frames"] + "A" * 100_000
        bad_url = (
            "frames-pack:base64,"
            + base64.b64encode(json.dumps(envelope).encode()).decode()
        )
        with self.assertRaises(ValueError) as caught:
            parse_d3_video_group_url(bad_url)
        self.assertIn("over the", str(caught.exception))

    def test_video_key_length_boundary_is_64(self):
        # The producer's video_key is a 64-char sha256 hex hash: 64 chars
        # round-trip, 65 are rejected on BOTH sides. The bound keeps every
        # identifying field (video_key, group_index, total_groups) inside the
        # engine media cache key's first 256 encoded chars — a longer key
        # could push group_index outside the truncated prefix.
        import base64
        import json

        frames = bytes(2 * 3 * FRAME * FRAME_W)
        url = encode_d3_video_group_url(
            "k" * 64, 0, 4, 24.0, [0, 1], FRAME, FRAME_W, GROUP_TOKENS, frames
        )
        self.assertEqual(parse_d3_video_group_url(url)["video_key"], "k" * 64)
        with self.assertRaises(ValueError):
            encode_d3_video_group_url(
                "k" * 65, 0, 4, 24.0, [0, 1], FRAME, FRAME_W, GROUP_TOKENS, frames
            )
        envelope = json.loads(
            base64.b64decode(url[len("frames-pack:base64,") :]).decode("utf-8")
        )
        envelope["video_key"] = "k" * 65
        bad_url = (
            "frames-pack:base64,"
            + base64.b64encode(json.dumps(envelope).encode()).decode()
        )
        with self.assertRaises(ValueError):
            parse_d3_video_group_url(bad_url)

    def test_cache_key_prefix_distinguishes_every_group(self):
        """The C++ engine media cache keys on ``url.substr(0, 256)``.

        The envelope puts every identifying field (video_key, group_index,
        total_groups, fps, frames_indices, height, width, tokens) BEFORE the
        frames payload, so the truncated key still identifies exactly one
        group of one clip; two groups never collide and different clips never
        collide (video_key is the content hash of the source video).
        """

        def prefix(url):
            return url[:256]

        urls = []
        for group in range(16):
            raw = bytes(group_frames(seed=group).numpy().tobytes())
            urls.append(
                encode_d3_video_group_url(
                    "clip-a",
                    group,
                    16,
                    24.0,
                    [2 * group, 2 * group + 1],
                    FRAME,
                    FRAME_W,
                    GROUP_TOKENS,
                    frames=raw,
                )
            )
        for group in range(16):
            raw = bytes(group_frames(seed=100 + group).numpy().tobytes())
            urls.append(
                encode_d3_video_group_url(
                    "clip-b",
                    group,
                    16,
                    30.0,
                    [2 * group, 2 * group + 1],
                    FRAME,
                    FRAME_W,
                    GROUP_TOKENS,
                    frames=raw,
                )
            )
        prefixes = [prefix(url) for url in urls]
        self.assertEqual(len(prefixes), len(set(prefixes)))
        # The same group of the same clip always yields the same key prefix
        # (deterministic dedup + cross-request reuse).
        raw = bytes(group_frames(seed=0).numpy().tobytes())
        again = encode_d3_video_group_url(
            "clip-a", 0, 16, 24.0, [0, 1], FRAME, FRAME_W, GROUP_TOKENS, frames=raw
        )
        self.assertEqual(prefix(again), prefixes[0])

    def test_multimodal_input_pickle_roundtrip_keeps_url_and_tensor(self):
        frames = group_frames(seed=13)
        url = self._encode(bytes(frames.numpy().tobytes()), group_index=1)
        mm_input = MultimodalInput(url, MMUrlType.VIDEO, frames, MMPreprocessConfig())
        restored = pickle.loads(pickle.dumps(mm_input))
        self.assertEqual(restored.url, url)
        self.assertEqual(restored.mm_type, MMUrlType.VIDEO)
        self.assertTrue(torch.equal(restored.tensor, frames))
        parsed = parse_d3_video_group_url(restored.url)
        self.assertEqual(parsed["group_index"], 1)
        self.assertEqual(parsed["frames"], bytes(frames.numpy().tobytes()))
        self.assertEqual(parsed["tokens"], GROUP_TOKENS)


class D3VideoGroupPositionsTest(TestCase):
    """Engine-composed positions == the official 5.17 get_rope_index output."""

    def compose(self, token_types, spans):
        return engine_generate_position_ids(token_types, spans)

    def test_single_video_matches_the_official_full_sequence(self):
        types, spans = d3_sequence(video_grids=[[2, 4, 4]], text_gaps=(11, 6, 13))
        official = official_get_rope_index(types, [], [[2, 4, 4]])
        engine = self.compose(types, spans)
        self.assertEqual(engine.shape, (len(types), 3))
        self.assertTrue(
            torch.equal(engine.t().to(torch.int64), official),
            "engine composition differs from the official get_rope_index",
        )

    def test_wide_frames_advance_by_max_side(self):
        # 64x96 frames: merged grid 2x3, spans advance current_pos by
        # max(4, 6) // 2 = 3 per group.
        types, spans = d3_sequence(video_grids=[[3, 4, 6]], text_gaps=(3, 4, 5))
        official = official_get_rope_index(types, [], [[3, 4, 6]])
        engine = self.compose(types, spans)
        self.assertTrue(torch.equal(engine.t().to(torch.int64), official))

    def test_mixed_image_and_video_matches_the_official_sequence(self):
        grid = [2, 4, 4]
        types, spans = d3_sequence(
            image_grid=[1, 4, 4], video_grids=[grid], text_gaps=(5, 2, 3, 8)
        )
        official = official_get_rope_index(types, [[1, 4, 4]], [grid])
        engine = self.compose(types, spans)
        self.assertTrue(torch.equal(engine.t().to(torch.int64), official))

    def test_repeated_video_matches_the_official_sequence(self):
        grid = [2, 4, 4]
        types, spans = d3_sequence(video_grids=[grid, grid], text_gaps=(4, 3, 2, 7))
        official = official_get_rope_index(types, [], [grid, grid])
        engine = self.compose(types, spans)
        self.assertTrue(torch.equal(engine.t().to(torch.int64), official))

    def test_local_positions_are_the_official_span_local_part(self):
        # What the mixin returns per group (t=0, h, w) is exactly the official
        # get_vision_position_ids with start_position=0 — the engine adds the
        # span base on top. Layout: the official port is component-major
        # [3, gh*gw] (one row per t/h/w component, modeling_qwen3_5.py's
        # stack(dim=0).reshape(3, -1)); the mixin's spans are token-major
        # [gh*gw, 3] (one row per token) — so exactly ONE transpose belongs in
        # the comparison. (The canonical d3_final_merged_units failure here was
        # an extra .t() on the official side — [3, N] vs [N, 3] — a size
        # mismatch, not a value or dtype difference: torch.equal does not
        # differentiate input dtypes.)
        for grid in ([1, 4, 4], [1, 4, 6], [1, 2, 8]):
            with self.subTest(grid=grid):
                official = official_get_vision_position_ids(0, grid, MERGE)
                local = local_positions(grid).t()
                self.assertEqual(
                    tuple(local.shape),
                    tuple(official.shape),
                    f"local {tuple(local.shape)} vs official {tuple(official.shape)}",
                )
                self.assertTrue(
                    torch.equal(local, official),
                    f"local {local.tolist()} vs official {official.tolist()}",
                )

    def test_group_metadata_never_enters_positions(self):
        # Positions depend only on the sequence and the span grids: computing
        # the groups in a different order (a cache miss after a hit, reordered
        # batches) yields byte-identical spans.
        grid = [2, 4, 4]
        types, spans = d3_sequence(video_grids=[grid], text_gaps=(9, 4, 6))
        reordered = [
            (start, local.clone())
            for start, local in sorted(spans, key=lambda span: -span[0])
        ]
        reordered.sort(key=lambda span: span[0])
        first = self.compose(types, spans)
        second = self.compose(types, reordered)
        self.assertTrue(torch.equal(first, second))


class D3VideoGroupEmbeddingTest(TestCase):
    """The real Qwen3.5 vision tower: per-group independence and positions."""

    @classmethod
    def setUpClass(cls):
        cls.config = Qwen3_5MoeVisionConfig(
            depth=2,
            hidden_size=48,
            hidden_act="gelu_pytorch_tanh",
            intermediate_size=96,
            num_heads=4,
            in_channels=3,
            patch_size=PATCH,
            spatial_merge_size=MERGE,
            temporal_patch_size=2,
            out_hidden_size=32,
            num_position_embeddings=2304,
        )
        cls.config._attn_implementation = "eager"
        params = VitParameters()
        params.config["ckpt_path"] = "unused-local-test-config"
        with patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin.AutoProcessor.from_pretrained",
            return_value=Mock(),
        ), patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin.Qwen2VLImageProcessor.from_pretrained",
            return_value=Mock(),
        ), patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin.Qwen3_5MoeVisionConfig.from_pretrained",
            return_value=cls.config,
        ), torch.random.fork_rng(
            devices=[]
        ):
            torch.manual_seed(7)
            cls.embedding = Qwen3_5MoeImageEmbedding(params)
        cls.embedding.visual.eval()

    def group_data(self, group_index, total_groups, seed, height=FRAME, width=FRAME_W):
        """One preprocessed d3 group: patches, grid, envelope metadata."""
        generator = torch.Generator().manual_seed(seed)
        patches = torch.randn(
            (height // PATCH) * (width // PATCH),
            2 * PATCH * PATCH * 3,
            generator=generator,
        )
        grid = torch.tensor([[1, height // PATCH, width // PATCH]], dtype=torch.int64)
        return (
            patches,
            grid,
            make_meta(group_index, total_groups, height=height, width=width),
        )

    def test_group_alone_equals_the_canonical_whole_clip(self):
        total = 4
        groups = [self.group_data(g, total, 100 + g) for g in range(total)]
        all_patches = torch.cat([data[0] for data in groups], dim=0)
        whole_grid = torch.tensor(
            [[total, FRAME // PATCH, FRAME_W // PATCH]], dtype=torch.int64
        )
        with torch.inference_mode():
            whole = self.embedding.visual(all_patches, grid_thw=whole_grid)
        whole = whole.pooler_output
        self.assertEqual(
            whole.shape, (total * GROUP_TOKENS, self.config.out_hidden_size)
        )
        for g, data in enumerate(groups):
            with torch.inference_mode():
                alone = self.embedding.visual(data[0], grid_thw=data[1]).pooler_output
            start = g * GROUP_TOKENS
            self.assertTrue(
                torch.equal(alone, whole[start : start + GROUP_TOKENS]),
                f"group {g} differs from its whole-clip rows",
            )

    def test_embedding_single_group_is_local_and_whole_clip_equal(self):
        total = 4
        groups = [self.group_data(g, total, 200 + g) for g in range(total)]
        all_patches = torch.cat([data[0] for data in groups], dim=0)
        whole_grid = torch.tensor([[total, FRAME // PATCH, FRAME_W // PATCH]])
        with torch.inference_mode():
            whole = self.embedding.visual(
                all_patches, grid_thw=whole_grid
            ).pooler_output
        for g, data in enumerate(groups):
            # embedding() is the per-item path: a cache miss of exactly one
            # group must reproduce its whole-clip embedding rows, with LOCAL
            # positions (the engine adds the span base).
            embeds, positions = self.embedding.embedding(data)
            start = g * GROUP_TOKENS
            self.assertTrue(torch.equal(embeds, whole[start : start + GROUP_TOKENS]))
            self.assertTrue(
                torch.equal(
                    positions, local_positions([1, FRAME // PATCH, FRAME_W // PATCH])
                )
            )

    def test_batches_splits_reorders_and_duplicates_do_not_change_groups(self):
        total = 5
        groups = [self.group_data(g, total, 300 + g) for g in range(total)]
        alone = [self.embedding.embedding(data) for data in groups]
        image = self.group_data(0, 1, 999)  # any [1, h, w] row behaves like an image
        image = (image[0], image[1])
        batches = [
            # all groups together, in order
            ([*groups], [MMUrlType.VIDEO] * total),
            # reordered
            ([groups[3], groups[0], groups[4]], [MMUrlType.VIDEO] * 3),
            # split: a single group alone in its own batch
            ([groups[1]], [MMUrlType.VIDEO]),
            # mixed with an image, groups interleaved
            (
                [image, groups[2], groups[0]],
                [MMUrlType.IMAGE, MMUrlType.VIDEO, MMUrlType.VIDEO],
            ),
            # the same group twice (duplicate video entries in one request)
            ([groups[4], groups[4]], [MMUrlType.VIDEO, MMUrlType.VIDEO]),
        ]
        for data_list, types in batches:
            with self.subTest(types=types):
                results = self.embedding.batched_embedding(data_list, types)
                self.assertEqual(len(results), len(data_list))
                for data, result in zip(data_list, results):
                    meta = d3_video_group_meta(data)
                    if meta is None:
                        continue
                    g = meta["group_index"]
                    self.assertTrue(torch.equal(result[0], alone[g][0]))
                    self.assertTrue(torch.equal(result[1], alone[g][1]))

    def test_partial_cache_hit_replays_cached_groups_exactly(self):
        # A hit replays the stored result; a miss recomputes alone. Both paths
        # equal the group's whole-clip rows, so any hit/miss mixture assembles
        # the identical full-video output.
        total = 4
        groups = [self.group_data(g, total, 400 + g) for g in range(total)]
        all_patches = torch.cat([data[0] for data in groups], dim=0)
        whole_grid = torch.tensor([[total, FRAME // PATCH, FRAME_W // PATCH]])
        with torch.inference_mode():
            whole = self.embedding.visual(
                all_patches, grid_thw=whole_grid
            ).pooler_output
        cached = {}
        assembled = [None] * total
        for g, data in enumerate(groups):
            if g % 2 == 0:  # hit: reuse the stored (alone-computed) result
                if g not in cached:
                    cached[g] = self.embedding.embedding(data)
                assembled[g] = cached[g]
            else:  # miss: computed fresh, alone
                assembled[g] = self.embedding.embedding(data)
        for g in range(total):
            start = g * GROUP_TOKENS
            self.assertTrue(
                torch.equal(assembled[g][0], whole[start : start + GROUP_TOKENS])
            )


class _RecordingVideoProcessor:
    """Records the exact call kwargs the consumer passes to the processor."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self._inner(**kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def build_recording_video_processor():
    """A real Qwen3VLVideoProcessor behind a call-recording wrapper."""
    from transformers import Qwen3VLVideoProcessor

    inner = Qwen3VLVideoProcessor(
        patch_size=PATCH,
        temporal_patch_size=2,
        merge_size=MERGE,
        size={"shortest_edge": 4096, "longest_edge": 200_704 * 32},
    )
    return _RecordingVideoProcessor(inner)


class D3VideoGroupPreprocessTest(TestCase):
    """The consumer preprocess: patchify pre-resized frames, never resample/resize."""

    @classmethod
    def setUpClass(cls):
        cls.video_processor = build_recording_video_processor()
        cls.processor = SimpleNamespace(video_processor=cls.video_processor)

    def group_input(self, frames):
        url = encode_d3_video_group_url(
            "clip",
            1,
            4,
            24.0,
            [2, 3],
            FRAME,
            FRAME_W,
            GROUP_TOKENS,
            frames=bytes(frames.contiguous().numpy().tobytes()),
        )
        return MultimodalInput(url, MMUrlType.VIDEO, frames, MMPreprocessConfig())

    def test_patchifies_the_pre_resized_group_without_resampling_or_resizing(self):
        from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import (
            Qwen3_VLImageEmbedding,
        )

        frames = torch.zeros(2, 3, FRAME, FRAME_W, dtype=torch.uint8)
        frames[:, :, :32, :32] = 255
        mm_input = self.group_input(frames)
        # The producer's frames are final: the consumer always passes the URL
        # payload's frames (byte-exact with the producer's tensor — the URL is
        # authoritative) and turns BOTH resampling and resizing off explicitly —
        # the 5.2 and 5.17 processors default do_sample_frames and do_resize
        # to True (left on, the bare 2-frame tensor would re-run the
        # fps/min_frames index math).
        #
        # The recording processor is shared class-wide, so the assertion is per
        # INVOCATION (exactly one new call each, with the right kwargs), never
        # a magic aggregate count. Every input form is covered: the plain
        # in-process tensor and the URL-fallback variants (the attached tensor
        # is not integrity evidence on any path — the RPC wire restores a
        # 1-element placeholder).
        inputs = [
            ("valid input", mm_input),
            (
                "rpc placeholder",
                MultimodalInput(
                    mm_input.url, MMUrlType.VIDEO, torch.empty(1), MMPreprocessConfig()
                ),
            ),
            (
                "wrong size",
                MultimodalInput(
                    mm_input.url,
                    MMUrlType.VIDEO,
                    torch.zeros(2, 3, 32, 32, dtype=torch.uint8),
                    MMPreprocessConfig(),
                ),
            ),
            (
                "wrong dtype",
                MultimodalInput(
                    mm_input.url,
                    MMUrlType.VIDEO,
                    torch.zeros(2, 3, FRAME, FRAME_W, dtype=torch.float32),
                    MMPreprocessConfig(),
                ),
            ),
            (
                "wrong rank",
                MultimodalInput(
                    mm_input.url,
                    MMUrlType.VIDEO,
                    torch.zeros(3, FRAME, FRAME_W, dtype=torch.uint8),
                    MMPreprocessConfig(),
                ),
            ),
            (
                "same shape, other bytes",
                MultimodalInput(
                    mm_input.url,
                    MMUrlType.VIDEO,
                    torch.full((2, 3, FRAME, FRAME_W), 7, dtype=torch.uint8),
                    MMPreprocessConfig(),
                ),
            ),
        ]
        results = {}
        for name, variant in inputs:
            with self.subTest(input=name):
                calls = self.processor.video_processor.calls
                calls.clear()
                with torch.inference_mode():
                    pixel_values, grid, meta = (
                        Qwen3_VLImageEmbedding._preprocess_d3_video_group(
                            variant, self.processor
                        )
                    )
                # Exactly ONE processor call for THIS invocation.
                self.assertEqual(len(calls), 1)
                call = calls[0]
                self.assertIs(call["do_resize"], False)
                self.assertIs(call["do_sample_frames"], False)
                # The URL payload's frames, byte-exact, whatever the input form.
                self.assertTrue(torch.equal(call["videos"][0], frames))
                results[name] = pixel_values
        # Every input form preprocesses to the same URL pixels.
        for name, pixel_values in results.items():
            with self.subTest(pixels=name):
                self.assertTrue(torch.equal(pixel_values, results["valid input"]))
        pixel_values = results["valid input"]
        # One temporal patch of a 4x4 patch grid: 16 patches of 2*16*16*3.
        self.assertEqual(tuple(pixel_values.shape), (16, 2 * PATCH * PATCH * 3))
        self.assertEqual(grid.tolist(), [[1, FRAME // PATCH, FRAME_W // PATCH]])
        self.assertEqual(meta["group_index"], 1)
        self.assertEqual(meta["tokens"], GROUP_TOKENS)
        # rescale + normalize of uint8 0/255 frames: exactly -1.0 and 1.0, and
        # the 32x32 corner of each frame fills the 2x2 corner patches.
        ones = (pixel_values == 1.0).sum().item()
        self.assertEqual(ones, 2 * 4 * PATCH * PATCH * 3)
        self.assertEqual(
            (pixel_values == -1.0).sum().item(),
            pixel_values.numel() - ones,
        )

    def test_any_attached_tensor_yields_the_url_frames(self):
        """The URL payload is authoritative; the attached tensor is ignored.

        The python RPC client never writes MultimodalInput.tensor into
        MultimodalInputPB and the C++ restore substitutes a 1-element
        placeholder (rtp_llm/cpp/model_rpc/QueryConverter.cc, the observed
        PRO5000 ``got (1,)`` case) — and on the in-process path a same-shaped
        tensor is not integrity evidence either — so the frames ALWAYS come
        from the validated URL payload, whatever the attached tensor is.
        """
        from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import (
            Qwen3_VLImageEmbedding,
        )

        frames = group_frames(seed=21)
        mm_input = self.group_input(frames)
        with torch.inference_mode():
            direct, _, _ = Qwen3_VLImageEmbedding._preprocess_d3_video_group(
                mm_input, self.processor
            )
        self.processor.video_processor.calls.clear()
        attached = [
            torch.empty(1),  # the exact QueryConverter placeholder
            torch.zeros(2, 3, 32, 32, dtype=torch.uint8),  # wrong size
            torch.zeros(2, 3, FRAME, FRAME_W, dtype=torch.float32),  # wrong dtype
            torch.zeros(3, FRAME, FRAME_W, dtype=torch.uint8),  # wrong rank
            # same shape and dtype, different values — not integrity evidence
            torch.full((2, 3, FRAME, FRAME_W), 7, dtype=torch.uint8),
        ]
        for tensor in attached:
            with self.subTest(shape=tuple(tensor.shape)):
                restored = MultimodalInput(
                    mm_input.url, MMUrlType.VIDEO, tensor, MMPreprocessConfig()
                )
                with torch.inference_mode():
                    via_url, grid, meta = (
                        Qwen3_VLImageEmbedding._preprocess_d3_video_group(
                            restored, self.processor
                        )
                    )
                # The URL payload is authoritative: byte-identical
                # preprocessing whatever the attached tensor is.
                self.assertTrue(torch.equal(via_url, direct))
                self.assertEqual(grid.tolist(), [[1, FRAME // PATCH, FRAME_W // PATCH]])
                self.assertEqual(meta["group_index"], 1)

    def test_url_payload_is_authoritative_over_a_same_shaped_tensor(self):
        """A same-shaped tensor with different bytes must not override the URL.

        The URL fully determines a d3 group (the engine media cache keys on
        it), so a same-shaped attached tensor can never fork the pixels: the
        result is the URL's pixels — equal to the RPC placeholder path — and
        NOT the wrong tensor's pixels.
        """
        from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import (
            Qwen3_VLImageEmbedding,
        )

        frames = group_frames(seed=22)
        mm_input = self.group_input(frames)
        wrong = torch.full((2, 3, FRAME, FRAME_W), 200, dtype=torch.uint8)
        # The wrong tensor's own pixels (its own URL): the counter-reference.
        wrong_url = encode_d3_video_group_url(
            "clip",
            1,
            4,
            24.0,
            [2, 3],
            FRAME,
            FRAME_W,
            GROUP_TOKENS,
            frames=bytes(wrong.contiguous().numpy().tobytes()),
        )
        with torch.inference_mode():
            via_placeholder, _, _ = Qwen3_VLImageEmbedding._preprocess_d3_video_group(
                MultimodalInput(
                    mm_input.url, MMUrlType.VIDEO, torch.empty(1), MMPreprocessConfig()
                ),
                self.processor,
            )
            via_wrong_tensor, _, _ = Qwen3_VLImageEmbedding._preprocess_d3_video_group(
                MultimodalInput(
                    mm_input.url, MMUrlType.VIDEO, wrong, MMPreprocessConfig()
                ),
                self.processor,
            )
            wrong_reference, _, _ = Qwen3_VLImageEmbedding._preprocess_d3_video_group(
                MultimodalInput(
                    wrong_url, MMUrlType.VIDEO, torch.empty(1), MMPreprocessConfig()
                ),
                self.processor,
            )
        # The URL's pixels on both paths: the attached tensor never overrides.
        self.assertTrue(torch.equal(via_wrong_tensor, via_placeholder))
        # And they are NOT the wrong tensor's pixels (same shape, other bytes).
        self.assertFalse(torch.equal(via_wrong_tensor, wrong_reference))

    def test_legacy_video_urls_still_take_the_legacy_path(self):
        # A plain data URL is not a d3 envelope and never enters the new branch.
        self.assertFalse(is_d3_video_group_url("data:video/mp4;base64,AAAA"))


class D3VideoGroupRpcTransportTest(TestCase):
    """The real frontend -> engine RPC serialization boundary for d3 groups.

    Exercises the PRODUCTION path pieces: the real serializer
    (``model_rpc_client.trans_input``, exactly what ``batch_generate`` calls
    per input), the real batch message (``BatchGenerateInputPB``), a
    SerializeToString/FromString wire round trip (the same bytes gRPC would
    carry), and the exact C++ restore semantics of ``QueryConverter.cc``
    (each ``MultimodalInputPB`` becomes a ``MultimodalInput`` whose tensor is
    the 1-element placeholder — the observed PRO5000 ``got (1,)`` case).
    The frames must survive all of that through the URL payload and preprocess
    to the URL payload's pixels — the URL is authoritative on every path (the
    attached tensor is ignored, placeholder or not).
    """

    @classmethod
    def setUpClass(cls):
        cls.processor = SimpleNamespace(
            video_processor=build_recording_video_processor()
        )

    def make_row(self, request_id, url, tensor, token_ids):
        from rtp_llm.config.generate_config import GenerateConfig
        from rtp_llm.utils.base_model_datatypes import GenerateInput

        return GenerateInput(
            request_id=request_id,
            token_ids=torch.tensor(token_ids, dtype=torch.int32),
            mm_inputs=[
                MultimodalInput(url, MMUrlType.VIDEO, tensor, MMPreprocessConfig())
            ],
            generate_config=GenerateConfig(max_new_tokens=0, is_streaming=False),
        )

    @staticmethod
    def restore_like_query_converter(mm_input_pb):
        """The C++ server-side restore, exactly as QueryConverter.cc builds it:
        url + a 1-element placeholder tensor + the config fields (the python
        client never transports the tensor, so the C++ side substitutes
        ``torch::empty(1)``)."""
        config = mm_input_pb.mm_preprocess_config
        return MultimodalInput(
            mm_input_pb.multimodal_url,
            MMUrlType.VIDEO,
            torch.empty(1),
            MMPreprocessConfig(
                width=config.width,
                height=config.height,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                fps=config.fps,
                min_frames=config.min_frames,
                max_frames=config.max_frames,
                crop_positions=list(config.crop_positions),
                mm_timeout_ms=config.mm_timeout_ms,
                max_long_side_pixel=config.max_long_side_pixel,
            ),
        )

    def test_group_frames_survive_the_real_rpc_wire_per_row(self):
        from rtp_llm.cpp.model_rpc.model_rpc_client import trans_input
        from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
            BatchGenerateInputPB,
        )
        from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import (
            Qwen3_VLImageEmbedding,
        )

        frames = group_frames(seed=31)
        raw = bytes(frames.contiguous().numpy().tobytes())
        url = encode_d3_video_group_url(
            "clip", 0, 4, 24.0, [0, 1], FRAME, FRAME_W, GROUP_TOKENS, frames=raw
        )
        # The producer construction (frontend_worker._d3_rows) attaches the
        # tensor AND the self-contained URL.
        rows = [
            self.make_row(101, url, frames, [11, 12, 13]),
            self.make_row(102, url, frames, [21, 22, 23, 24]),
        ]
        # The real production serializer (batch_generate builds exactly this).
        batch_pb = BatchGenerateInputPB()
        for row in rows:
            batch_pb.inputs.append(trans_input(row))
        # The REAL boundary behavior: the pb mm input carries url/type/config
        # and NO tensor — this is why the frames must ride inside the URL.
        for input_pb in batch_pb.inputs:
            pb_mm = input_pb.multimodal_inputs[0]
            self.assertEqual(pb_mm.multimodal_url, url)
            self.assertFalse(pb_mm.HasField("multimodal_tensor"))
        # The gRPC-equivalent wire round trip.
        parsed = BatchGenerateInputPB.FromString(batch_pb.SerializeToString())
        self.assertEqual(len(parsed.inputs), len(rows))
        # The reference result: the URL payload is authoritative on every path
        # (attached tensor or C++ placeholder — both yield the same pixels).
        direct, _, _ = Qwen3_VLImageEmbedding._preprocess_d3_video_group(
            MultimodalInput(url, MMUrlType.VIDEO, frames, MMPreprocessConfig()),
            self.processor,
        )
        # Every row restores through the exact C++ semantics: the placeholder
        # tensor is replaced by the URL payload frames, byte-identically, and
        # both rows share the byte-identical group URL (the engine media cache
        # dedups by URL), so rows stay independent and duplicates are free.
        for row_pb in parsed.inputs:
            restored = self.restore_like_query_converter(row_pb.multimodal_inputs[0])
            self.assertEqual(
                tuple(restored.tensor.shape), (1,), "the placeholder must be (1,)"
            )
            via_rpc, grid, meta = Qwen3_VLImageEmbedding._preprocess_d3_video_group(
                restored, self.processor
            )
            self.assertTrue(
                torch.equal(via_rpc, direct),
                "the URL frames must preprocess identically on every path",
            )
            self.assertEqual(grid.tolist(), [[1, FRAME // PATCH, FRAME_W // PATCH]])
            # The returned meta is pool-lean: the transport-only frames bytes
            # are dropped (they would dominate the pool pickle for nothing).
            self.assertNotIn("frames", meta)
            self.assertEqual(meta["group_index"], 0)
            # The URL itself still carries the exact producer bytes (a later
            # cache miss of the same group replays them deterministically).
            self.assertEqual(parse_d3_video_group_url(url)["frames"], raw)
        # The explicit no-resample/no-resize contract held on every call.
        for call in self.processor.video_processor.calls:
            self.assertIs(call["do_resize"], False)
            self.assertIs(call["do_sample_frames"], False)


if __name__ == "__main__":
    main()
