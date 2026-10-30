import os
from pathlib import Path
import struct
import subprocess

import pytest
import zstandard

from mock_framework.input import MAGIC, MAX_FRAME, MAX_BATCH_REQUESTS, read_inputs, write_inputs, pb

ROOT = Path(__file__).resolve().parents[1]


def requests(count=2050):
    for i in range(count):
        request = pb.Request(token_ids=[i], max_new_tokens=3, arrival_offset_ms=i * 7)
        if i % 2 == 0:
            request.priority = i % 100
        yield request


def frame(batch, *, checksum=True):
    encoded = zstandard.ZstdCompressor(write_checksum=checksum).compress(batch.SerializeToString())
    return struct.pack('>I', len(encoded)) + encoded


def test_round_trip_multiple_batches_and_optional_zero_priority(tmp_path):
    path = tmp_path / 'input.binpb.zst'
    write_inputs(path, requests())
    assert list(read_inputs(path)) == list(requests())
    assert next(read_inputs(path)).HasField('priority')
    with pytest.raises(FileExistsError):
        write_inputs(path, [])
    assert len(list(read_inputs(path))) == 2050


def test_writer_removes_failed_output_without_publishing_a_valid_prefix(tmp_path):
    def broken():
        yield from requests(1)
        raise RuntimeError('input preparation failed')
    path = tmp_path / 'input.binpb.zst'
    with pytest.raises(RuntimeError, match='preparation failed'):
        write_inputs(path, broken())
    assert not path.exists()


@pytest.mark.parametrize('input_request', [pb.Request(max_new_tokens=3), pb.Request(token_ids=[-1], max_new_tokens=3),
    pb.Request(token_ids=[1], max_new_tokens=0), pb.Request(token_ids=[1], max_new_tokens=3, priority=-1),
    pb.Request(token_ids=[1], max_new_tokens=1, arrival_offset_ms=-1),
    pb.Request.FromString(pb.Request(token_ids=[1], max_new_tokens=3).SerializeToString() + b'\xa0\x06\x01')])
def test_invalid_or_unknown_fields_are_rejected_by_writer_and_reader(tmp_path, input_request):
    path = tmp_path / 'bad.binpb.zst'
    with pytest.raises(ValueError):
        write_inputs(path, [input_request])
    assert not path.exists()
    path.write_bytes(MAGIC + frame(pb.Batch(sequence=1, requests=[input_request])))
    with pytest.raises(ValueError):
        list(read_inputs(path))


@pytest.mark.parametrize('damage', ['legacy-json', 'truncated', 'missing-seal', 'bad-count', 'sequence', 'checksum',
                                  'oversized-length', 'oversized-content', 'no-checksum', 'trailing', 'too-many', 'extra-frame'])
def test_bad_input_cannot_be_mistaken_for_a_complete_file(tmp_path, damage):
    request = next(requests())
    data = frame(pb.Batch(sequence=1, requests=[request]))
    seal = frame(pb.Batch(sequence=2, total_requests=1))
    content = MAGIC + data + seal
    if damage == 'legacy-json': content = b'{"token_ids":[1],"max_new_tokens":3}\n'
    if damage == 'truncated': content = content[:-1]
    if damage == 'missing-seal': content = MAGIC + data
    if damage == 'bad-count': content = MAGIC + data + frame(pb.Batch(sequence=2, total_requests=2))
    if damage == 'sequence': content = MAGIC + frame(pb.Batch(sequence=2, requests=[request]))
    if damage == 'checksum': content = MAGIC + data[:-1] + bytes([data[-1] ^ 1]) + seal
    if damage == 'oversized-length': content = MAGIC + struct.pack('>I', MAX_FRAME + 1)
    if damage == 'oversized-content':
        encoded = zstandard.ZstdCompressor(write_checksum=True).compress(b'\0' * (MAX_FRAME + 1))
        content = MAGIC + struct.pack('>I', len(encoded)) + encoded
    if damage == 'no-checksum': content = MAGIC + frame(pb.Batch(sequence=1, requests=[request]), checksum=False) + seal
    if damage == 'trailing': content += b'extra'
    if damage == 'extra-frame':
        encoded = data[4:] + zstandard.ZstdCompressor(write_checksum=True).compress(b'')
        content = MAGIC + struct.pack('>I', len(encoded)) + encoded + seal
    if damage == 'too-many': content = MAGIC + frame(pb.Batch(sequence=1, requests=[request] * (MAX_BATCH_REQUESTS + 1)))
    path = tmp_path / 'bad.binpb.zst'
    path.write_bytes(content)
    with pytest.raises(ValueError):
        list(read_inputs(path))


def test_python_and_java_use_one_proto_and_the_same_framing(tmp_path):
    java_home = os.environ.get('JAVA_HOME')
    if not java_home:
        pytest.skip('Java 21 and built Client artifacts required')
    classpath = os.pathsep.join([str(ROOT / 'java/client/target/test-classes'),
        str(ROOT / 'java/client/target/mock-client-1.0-SNAPSHOT-all.jar')])
    prefix = [str(Path(java_home) / 'bin/java'), '-cp', classpath, 'org.flexlb.mock.client.InputProbe']
    python_file, java_file = tmp_path / 'python.binpb.zst', tmp_path / 'java.binpb.zst'
    write_inputs(python_file, requests())
    read = subprocess.run(prefix + ['read', str(python_file)], capture_output=True, text=True, timeout=15)
    assert read.returncode == 0, read.stderr
    write = subprocess.run(prefix + ['write', str(java_file)], capture_output=True, text=True, timeout=15)
    assert write.returncode == 0, write.stderr
    assert list(read_inputs(java_file)) == list(requests())


def test_image_spans_round_trip_and_reject_overlap(tmp_path):
    image = pb.Image(url='mock://image', width=56, height=28, offset=1, length=2)
    request = pb.Request(token_ids=[10, 0, 0, 20], max_new_tokens=3, images=[image])
    path = tmp_path / 'images.binpb.zst'
    write_inputs(path, [request])
    assert list(read_inputs(path)) == [request]
    request.images.add().CopyFrom(image)
    with pytest.raises(ValueError, match='span'):
        write_inputs(tmp_path / 'overlap.binpb.zst', [request])
