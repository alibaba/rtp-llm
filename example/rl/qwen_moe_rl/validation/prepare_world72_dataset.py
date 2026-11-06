"""Freeze 18 genuine DeepScaleR questions for two real world72 GRPO updates.

Keep the eight earlier accepted questions verbatim; append ten deterministically
selected official rows. Never include official solutions or answers in prompts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

SOURCE_SHA256 = '4e9e7b18248d982b4498a54a6bdb37b1bb653cf5f77bc309fa67a6c2533c8ba4'
EIGHT_SHA256 = '2642350d3158fb11da15589c3e848598efc09d6852e57cdd0311688c635339cf'
FIRST_EIGHT_INDICES = (39642,5900,10715,12346,21859,39638,20184,29526)
INSTRUCTION = '\n\nUse the Python tool at least once to check a calculation before giving the final boxed answer.'


def normalize(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(value)))


def prepare(source: Path, previous_eight: Path | None, output: Path):
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError('Official full DeepScaleR source SHA256 differs')
    rows = json.loads(payload)
    if len(rows) != 40315:
        raise ValueError('The official 40,315 source rows are required')
    selected = [dict(question=rows[index]['problem']+INSTRUCTION, ground_truth=str(rows[index]['answer']),
        data_source='deepscaler_math', source_row_index=index, original_question=rows[index]['problem'])
        for index in FIRST_EIGHT_INDICES]
    previous = ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected).encode()
    if hashlib.sha256(previous).hexdigest() != EIGHT_SHA256:
        raise ValueError('The reconstructed prior eight-question fixture SHA256 differs')
    if previous_eight is not None and previous_eight.read_bytes()!=previous:
        raise ValueError('The supplied prior accepted eight-question fixture differs')
    if len(selected) != 8:
        raise ValueError('The baseline fixture must contain exactly eight genuine source prompts')
    indices = set(); questions = set()
    for row in selected:
        index = row['source_row_index']; official = rows[index]
        if (row['original_question'] != official['problem'] or row['ground_truth'] != str(official['answer'])
                or row['question'] != official['problem'] + INSTRUCTION or row['data_source'] != 'deepscaler_math'):
            raise ValueError('The preserved baseline task does not reproduce its official source row')
        indices.add(index); questions.add(normalize(official['problem']))
    candidates = []
    for index, row in enumerate(rows):
        question = row['problem']; key = normalize(question)
        if index in indices or key in questions or not 180 <= len(question) <= 1400:
            continue
        candidates.append((index, row)); questions.add(key)
    random.Random(36).shuffle(candidates)
    for index, row in candidates[:10]:
        selected.append(dict(question=row['problem']+INSTRUCTION, ground_truth=str(row['answer']),
            data_source='deepscaler_math', source_row_index=index, original_question=row['problem']))
    if len(selected) != 18 or len({row['source_row_index'] for row in selected}) != 18:
        raise ValueError('Exactly 18 distinct source questions are required')
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in selected).encode()
    if output.exists() and output.read_bytes() != payload:
        raise FileExistsError('Refusing to overwrite a different frozen training input')
    output.write_bytes(payload)
    manifest = dict(schema='q38-world72-genuine-deepscaler18-v1', source_dataset='agentica-org/DeepScaleR-Preview-Dataset',
        official_revision='4500022f6cb0a1456ea5eabe2ff366d4610ee339', source_sha256=SOURCE_SHA256,
        source_rows=40315, source_bytes=source.stat().st_size, prior_eight_sha256=EIGHT_SHA256,
        selection_seed_for_ten_additional_rows=36, selected_source_indices=[row['source_row_index'] for row in selected],
        rows=18, output_sha256=hashlib.sha256(payload).hexdigest(), output_bytes=len(payload),
        trajectories_per_prompt=4, actual_trajectories_per_iteration=72, required_actual_updates=2,
        ppo_mini_batch_size_prompt_units=18, ppo_epochs=1, global_minibatches_per_update=1,
        reward='Original final boxed-answer correctness only; no tool bonus or synthetic rewards.',
        nonuniform_rewards='Must be observed in at least one same-question group in each actual iteration; not guaranteed by data selection.',
        thinking_enabled=True, training_evaluation_overlap_not_checked=True,
        scope='Functional RL path acceptance only; not a held-out model-quality benchmark.')
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--previous-eight',type=Path,help='Optional byte-exact check against the prior eight-question fixture; reconstructed from pinned official row indices by default')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.previous_eight,args.output)))
