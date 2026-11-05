"""Read the pinned Qwen checkpoint identity and tensor shapes without allocating tensors."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import struct

MODEL_ID = 'Qwen/Qwen3.5-397B-A17B'
REVISION = '8472618112abcbd45acbcdc58436aff4233c23f7'
TEXT_NUMEL = 396346350336
VISUAL_NUMEL = 456010480
PINNED_HASHES = {
    'config.json': '3ae7fa89c2f7d1354096418ddaf1331e9e0898a8ca11e804fb6dbaa087efb7da',
    'model.safetensors.index.json': '407d6a184a29469034ad92bcfaa6b72b582f8ad6afcbfad18a38a1c59ed296f8',
}


def default_model(project):
    return Path(project)/'models'/MODEL_ID.split('/')[1]


def metadata(snapshot):
    documents = {}
    for filename, expected in PINNED_HASHES.items():
        payload = (snapshot/filename).read_bytes()
        if sha256(payload).hexdigest() != expected:
            raise ValueError(f'{filename} differs from {MODEL_ID}@{REVISION}')
        documents[filename] = json.loads(payload)
    return documents['config.json']['text_config'], documents['model.safetensors.index.json']


def identity():
    return dict(model_id=MODEL_ID, model_revision=REVISION, metadata_sha256=PINNED_HASHES.copy())


@dataclass
class Checkpoint:
    text: dict
    visual: dict
    mtp: dict
    gdn: set
    experts: set
    text_numel: int
    visual_numel: int
    mtp_numel: int
    num_layers: int
    num_experts: int
    initial_bf16_fingerprints: dict

    @property
    def active_bf16_bytes(self):
        return 2 * (self.text_numel + self.visual_numel)


def partition(name):
    if name == 'lm_head.weight' or name.startswith('model.language_model.'):
        return 'text'
    if name.startswith('model.visual.'):
        return 'visual'
    if name.startswith('mtp.'):
        return 'mtp'
    raise ValueError(f'Unexpected checkpoint tensor namespace {name}')


def shard_header(path):
    with path.open('rb') as src:
        header_bytes = src.read(8)
        if len(header_bytes) != 8:
            raise ValueError(f'{path}: missing safetensors header length')
        length, = struct.unpack('<Q', header_bytes)
        if not 2 <= length <= 100_000_000:
            raise ValueError(f'{path}: invalid safetensors header length {length}')
        header = json.loads(src.read(length))
    header.pop('__metadata__', None)
    return header, path.stat().st_size - (8 + length)


def load_checkpoint(snapshot):
    config, index = metadata(snapshot)
    weight_map = index['weight_map']
    groups = {'text': {}, 'visual': {}, 'mtp': {}}
    totals = {key: 0 for key in groups}
    total_data = 0
    layers = config['num_hidden_layers']
    sample_names = {
        f'model.language_model.layers.{layer}.{leaf}'
        for layer in (0, layers//2, layers-1)
        for leaf in ('input_layernorm.weight', 'mlp.shared_expert_gate.weight')
    }
    samples = {}
    for filename in sorted(set(weight_map.values())):
        path = snapshot/filename
        header, bytes_in_file = shard_header(path)
        expected = {name for name, shard in weight_map.items() if shard == filename}
        if set(header) != expected:
            raise ValueError(f'{filename}: shard keys differ from pinned checkpoint index')
        spans = []
        for name, entry in header.items():
            shape = entry['shape']
            if not shape or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError(f'{filename}: invalid shape for {name}')
            start, stop = entry['data_offsets']
            count = math.prod(shape)
            width = {'BF16': 2, 'F32': 4}[entry['dtype']]
            if start < 0 or stop - start != count*width:
                raise ValueError(f'{filename}: shape/byte extent disagree for {name}')
            spans.append((start, stop))
            group = partition(name)
            groups[group][name] = shape
            totals[group] += count
        position = 0
        for start, stop in sorted(spans):
            if start != position:
                raise ValueError(f'{filename}: invalid contiguous tensor extents')
            position = stop
        if position != bytes_in_file:
            raise ValueError(f'{filename}: shard size disagrees with header')
        total_data += position
        for name in sample_names & header.keys():
            entry = header[name]
            start, stop = entry['data_offsets']
            if entry['dtype'] != 'BF16' or not 0 < stop-start <= 128*1024:
                raise ValueError(f'{name}: unexpectedly large or non-BF16 initial weight fingerprint')
            with path.open('rb') as src:
                src.seek(path.stat().st_size - bytes_in_file + start)
                payload = src.read(stop-start)
            if len(payload) != stop-start:
                raise ValueError(f'{name}: incomplete initial fingerprint')
            samples[name] = sha256(payload).hexdigest()
    if samples.keys() != sample_names:
        raise ValueError('Incomplete checkpoint fingerprints')
    if total_data != index['metadata']['total_size']:
        raise ValueError('Checkpoint bytes disagree with the pinned index')
    layers, experts = config['num_hidden_layers'], config['num_experts']
    if (layers, experts, totals['text'], totals['visual'], len(groups['text']), len(groups['visual'])) != (
            60, 512, TEXT_NUMEL, VISUAL_NUMEL, 1038, 333):
        raise ValueError('Official checkpoint structure/element totals differ from the fixed 397B contract')
    layer_types = config['layer_types']
    if len(layer_types) != layers:
        raise ValueError('Attention layer_types and layer count disagree')
    text = groups['text']
    packed = {f'model.language_model.layers.{i}.mlp.experts.{leaf}' for i in range(layers)
              for leaf in ('gate_up_proj', 'down_proj')}
    special = {f'model.language_model.layers.{i}.linear_attn.{leaf}' for i, kind in enumerate(layer_types)
               if kind == 'linear_attention' for leaf in ('A_log', 'dt_bias')}
    if packed != {name for name in text if '.mlp.experts.' in name} or special != {
        name for name in text if name.endswith(('.linear_attn.A_log', '.linear_attn.dt_bias'))
    }:
        raise ValueError('Official layer list disagrees with packed expert or GDN names')
    inner, hidden = config['moe_intermediate_size'], config['hidden_size']
    for name in packed:
        expected_shape = [experts, 2*inner, hidden] if name.endswith('gate_up_proj') else [experts, hidden, inner]
        if text[name] != expected_shape:
            raise ValueError(f'{name}: packed expert shape differs from the config')
    return Checkpoint(text=text, visual=groups['visual'], mtp=groups['mtp'], gdn=special, experts=packed,
                      text_numel=totals['text'], visual_numel=totals['visual'], mtp_numel=totals['mtp'],
                      num_layers=layers, num_experts=experts, initial_bf16_fingerprints=samples)
