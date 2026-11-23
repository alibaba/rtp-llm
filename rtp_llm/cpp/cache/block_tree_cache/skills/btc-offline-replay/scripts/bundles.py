"""Read historical tar-packed BTCPFX01 datasets."""
import shutil
import tarfile
from pathlib import Path


def unpack(directory, chunks, dest):
    from itertools import groupby
    for name, group in groupby(chunks, key=lambda row: row["bundle"]):
        if Path(name).name != name:
            raise ValueError("invalid bundle name")
        with tarfile.open(directory / name, "r|") as archive:
            for row in group:
                member = archive.next()
                if member is None or member.name != row["file"] or not member.isfile():
                    raise ValueError("unexpected member in " + name)
                with archive.extractfile(member) as source, (dest / row["file"]).open("xb") as target:
                    shutil.copyfileobj(source, target, 1 << 20)


def legacy_view(data, chunks, dest, base, domain):
    """Stream physical files into the fixed-stride input view required by old runners."""
    import struct
    from contextlib import ExitStack
    name = domain.encode()
    dest.mkdir()
    written = in_chunk = tokens = 0
    with ExitStack() as handles:
        out = None
        for chunk in chunks:
            with (data / chunk["file"]).open("rb") as src:
                src.seek(32 + len(name))
                for _ in range(chunk["requests"]):
                    head = src.read(16)
                    _, length, id_size = struct.unpack("<QII", head)
                    if out is None:
                        out = handles.enter_context((dest / f"{written // 100000:06}.btcbin").open("w+b"))
                        out.write(struct.pack("<8sIQQI", b"BTCPFX01", base, 0, 0, len(name)) + name)
                        in_chunk = tokens = 0
                    size = id_size + 8 * (length // base)
                    body = src.read(size)
                    if len(body) != size:
                        raise ValueError("truncated BTCPFX01 record")
                    out.write(head)
                    out.write(body)
                    written += 1
                    in_chunk += 1
                    tokens += length
                    if in_chunk == 100000:
                        out.seek(12)
                        out.write(struct.pack("<QQ", in_chunk, tokens))
                        out.close()
                        out = None
                if src.read(1):
                    raise ValueError("unexpected trailing chunk data")
        if out is not None:
            out.seek(12)
            out.write(struct.pack("<QQ", in_chunk, tokens))
