import os

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.cp_cache_transport import (
    PackedRows,
    PagedRows,
    pack_rows,
    paged_row_map,
    restore_indexer,
    restore_mla,
)


def test_ragged_row_map_does_not_address_padding() -> None:
    actual = torch.tensor([0, 3, 1, 0], device="cuda", dtype=torch.int32)
    padded = torch.tensor([4, 4, 4, 0], device="cuda", dtype=torch.int32)
    expected = torch.tensor(
        [
            -1,
            -1,
            -1,
            -1,
            1 << 32,
            (1 << 32) + 1,
            (1 << 32) + 2,
            -1,
            2 << 32,
            -1,
            -1,
            -1,
        ],
        device="cuda",
    )
    torch.testing.assert_close(
        paged_row_map(actual, padded, 12), expected, rtol=0, atol=0
    )


def test_paged_pack_preserves_bytes_and_zeroes_padding() -> None:
    cache = torch.arange(7 * 3 * 656, device="cuda").to(torch.uint8).view(7, 3, 656)
    table = torch.tensor([[5, 2], [6, 1]], device="cuda", dtype=torch.int32)
    rows = torch.tensor([0, 4, -1, (1 << 32) + 2, (1 << 32) + 3], device="cuda")
    expected = torch.stack(
        (
            cache[5, 0],
            cache[2, 1],
            torch.zeros_like(cache[0, 0]),
            cache[6, 2],
            cache[1, 0],
        )
    )
    actual = pack_rows(PagedRows(cache, table, rows))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_indexer_pack_reads_block_major_scales() -> None:
    for head, scale in ((128, 4), (128, 16)):
        width, page = head + scale, 7
        cache = (
            torch.arange(5 * page * width, device="cuda")
            .to(torch.uint8)
            .view(5, page, width)
        )
        table = torch.tensor([[4, 2]], device="cuda", dtype=torch.int32)
        rows = torch.tensor([1, 8, -1], device="cuda")
        expected = torch.zeros((3, width), device="cuda", dtype=torch.uint8)
        raw = cache.view(5, -1)
        for row, block in ((0, 4), (1, 2)):
            expected[row, :head] = raw[block, head : 2 * head]
            expected[row, head:] = raw[
                block, page * head + scale : page * head + 2 * scale
            ]
        actual = pack_rows(PagedRows(cache, table, rows, head))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_joint_restore_is_bit_exact_and_preserves_unwritten_rows() -> None:
    torch.manual_seed(127)
    rows, ranks, width = 13, 3, 656
    quant = torch.randn(ranks * rows, 512, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.rand(ranks * rows, 4, device="cuda") * 4
    rope = torch.randn(ranks * rows, 64, device="cuda", dtype=torch.bfloat16)
    raw = torch.cat(
        (quant.view(torch.uint8), scale.view(torch.uint8), rope.view(torch.uint8)),
        dim=1,
    )
    prefix, suffix = 17, 29
    packet = torch.full(
        (ranks, prefix + rows * width + suffix), 255, device="cuda", dtype=torch.uint8
    )
    packet[:, prefix : prefix + rows * width] = raw.view(ranks, -1)
    mapping = torch.tensor([37, -1, 0, 19, 5], device="cuda")
    destinations = torch.tensor([6, 2, 7, 0, 4], device="cuda")
    out = torch.full((9, 576), -13, device="cuda", dtype=torch.bfloat16)
    expected = out.clone()
    decoded = torch.cat(
        (
            (quant.float().view(-1, 4, 128) * scale[:, :, None])
            .view(-1, 512)
            .bfloat16(),
            rope,
        ),
        dim=1,
    )
    expected[destinations[mapping >= 0]] = decoded[mapping[mapping >= 0]]
    view = PackedRows(packet, rows, width, packet.shape[1], prefix)
    restore_mla(view, mapping, out, destinations)
    torch.testing.assert_close(
        out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )


def test_joint_indexer_restore_keeps_all_scale_formats() -> None:
    for scales in (4, 16):
        head, rows, ranks, offset = 128, 9, 3, 23
        width = head + scales
        packet = torch.randint(
            0, 256, (ranks, offset + rows * width), device="cuda", dtype=torch.uint8
        )
        mapping = torch.tensor([26, 1, 9, 17], device="cuda")
        key = torch.empty((4, head), device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.empty((4, scales), device="cuda", dtype=torch.uint8)
        expected = packet[:, offset:].reshape(-1, width)[mapping]
        restore_indexer(
            PackedRows(packet, rows, width, packet.shape[1], offset),
            mapping,
            (key, scale),
        )
        torch.testing.assert_close(
            key.view(torch.uint8), expected[:, :head], rtol=0, atol=0
        )
        torch.testing.assert_close(scale, expected[:, head:], rtol=0, atol=0)


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    test_ragged_row_map_does_not_address_padding()
    test_paged_pack_preserves_bytes_and_zeroes_padding()
    test_indexer_pack_reads_block_major_scales()
    test_joint_restore_is_bit_exact_and_preserves_unwritten_rows()
    test_joint_indexer_restore_keeps_all_scale_formats()
    print("CP cache transport: five GPU cases passed")
