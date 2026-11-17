"""Fixed-image PD/MTP regression for the opt-in ATOM draft-loop graph.

Run once with the D-side graph flag disabled and once enabled. The test
requires existing TP2 P + TP2 D services, with P cache on and D cache off.
It creates deterministic image fixtures; no external image URLs are used.
"""
import argparse
import asyncio
import base64
import hashlib
import json
import re
from pathlib import Path

import aiohttp
from PIL import Image, ImageDraw, ImageFont


def fixtures(root):
    root.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=64)
    def save(name, size, draw):
        image = Image.new('RGB', size, 'white')
        draw(ImageDraw.Draw(image))
        image.save(root / (name + '.png'))
    save('red_square', (512, 512), lambda d: d.rectangle((100, 100, 412, 412), fill='red'))
    save('blue_square', (512, 512), lambda d: d.rectangle((100, 100, 412, 412), fill='blue'))
    save('three_circles', (768, 384), lambda d: [d.ellipse((40+i*240, 100, 200+i*240, 260), fill='green') for i in range(3)])
    save('ocr', (768, 384), lambda d: d.text((80, 140), 'BLUE 7429', font=font, fill='black'))
    save('wide', (1280, 384), lambda d: (d.rectangle((80, 100, 300, 300), fill='red'), d.ellipse((950, 100, 1150, 300), fill='blue')))
    save('tall', (384, 1280), lambda d: (d.rectangle((80, 100, 300, 320), fill='red'), d.ellipse((90, 950, 290, 1150), fill='blue')))
    save('yellow_triangle', (512, 512), lambda d: d.polygon([(256, 70), (70, 440), (440, 440)], fill='yellow'))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.png')}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {p.stem: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() for p in root.glob('*.png')}


def cases(images):
    def case(name, files, question, expected, long=False, multi_turn=False):
        content = [{'type': 'image_url', 'image_url': {'url': images[f]}} for f in files]
        content.append({'type': 'text', 'text': question})
        messages = [{'role': 'user', 'content': content}]
        if multi_turn:
            messages[0]['content'][-1]['text'] = 'Please inspect this image.'
            messages.extend([{'role': 'assistant', 'content': 'I have inspected it.'}, {'role': 'user', 'content': question}])
        return dict(name=name, messages=messages, expected=expected, long=long, images=len(files))
    color = 'What color is the square? Reply with one English color word.'
    return [
        case('single_red', ['red_square'], color, ['red']),
        case('changed_image_blue', ['blue_square'], color, ['blue']),
        case('ocr', ['ocr'], 'Read all the printed text. Output only that text.', [r'blue\s*7429']),
        case('count', ['three_circles'], 'How many green circles are there? Reply with a single digit.', ['3|three']),
        case('wide', ['wide'], 'Is the red square to the left or right of the blue circle? Reply only left or right.', ['left']),
        case('tall', ['tall'], 'Is the red square above or below the blue circle? Reply only above or below.', ['above']),
        case('shape', ['yellow_triangle'], 'Name the yellow shape. Reply with one word.', ['triangle']),
        case('multi_second_blue', ['red_square', 'blue_square'], 'What color is the square in the SECOND image? Reply with one color word.', ['blue']),
        case('multi_second_red', ['blue_square', 'red_square'], 'What color is the square in the SECOND image? Reply with one color word.', ['red']),
        case('multi_turn', ['red_square'], color, ['red'], multi_turn=True),
        case('wide_long', ['wide'], 'Describe the shapes, their colors, and their relative positions in detail. Explain how the layout differs from a vertical arrangement. Write at least 80 English words.', ['red', 'blue', 'left', 'right'], long=True),
    ]


def make_video_fixture(path):
    import av  # Optional dependency, only needed with --include-video.
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=4)
        stream.width, stream.height, stream.pix_fmt = 384, 256, 'yuv420p'
        for index in range(16):
            image = Image.new('RGB', (384, 256), 'white')
            ImageDraw.Draw(image).rectangle((100, 40, 280, 220), fill='red' if index < 8 else 'blue')
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def video_cases(path):
    """Use a fixed four-second clip: red square first, blue square last."""
    url = 'data:video/mp4;base64,' + base64.b64encode(path.read_bytes()).decode()
    result = []
    for name, question, long in [
        ('video_short', 'What are the colors of the square at the START and the END of the video, in that order? Reply with those two colors only.', False),
        ('video_long', 'Describe how the square changes during this video. Explain the initial and final appearance, the background, and what remains constant. Write at least 80 English words.', True),
    ]:
        result.append(dict(name=name, images=1, long=long,
            expected=[r'red[\s\S]*blue'] if not long else ['red', 'blue'],
            messages=[{'role': 'user', 'content': [
                {'type': 'video_url', 'video_url': {'url': url}},
                {'type': 'text', 'text': question}]}]))
    return result


async def request(session, base_url, case, cache, phase):
    body = {'model': 'default', 'messages': case['messages'], 'temperature': 0,
            'top_k': 1, 'max_completion_tokens': 192 if case['long'] else 48,
            'stream': True, 'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': False},
            'extra_configs': {'force_disable_sp_run': False, 'reuse_cache': cache}}
    text, aux, usage, reason, events = '', {}, {}, None, []
    async with session.post(base_url + '/v1/chat/completions', json=body) as response:
        if response.status != 200:
            raise RuntimeError(f'{case["name"]}: HTTP {response.status}: {await response.text()}')
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            event = json.loads(line[6:]); events.append(event)
            if event.get('error'):
                raise RuntimeError(str(event['error']))
            aux = event.get('aux_info') or aux
            usage = event.get('usage') or usage
            for choice in event.get('choices', []):
                text += (choice.get('delta') or {}).get('content') or ''
                reason = choice.get('finish_reason') or reason
    correct = all(re.search(r'\b(?:' + pattern + r')\b', text.lower()) for pattern in case['expected'])
    image_tokens = usage.get('prompt_tokens_details', {}).get('image_tokens', 0)
    # Some API versions report image lengths only in aux_info.
    image_seen = bool(image_tokens or any(int(v) > 0 for v in (aux.get('multimodal_lengths') or {}).values()))
    feature = aux.get('pd_sep') is True and reason in ('stop', 'length')
    feature = feature and (not case['images'] or image_seen)
    feature = feature and (cache or aux.get('reuse_len') == 0)
    if phase == 'cache_hit':
        feature = feature and aux.get('reuse_len', 0) >= 64
    if case['long']:
        feature = feature and aux.get('speculative_draft_rounds', 0) > 1 and usage.get('completion_tokens', 0) >= 32
    return dict(name=case['name'], phase=phase, correct=bool(correct), feature_ok=bool(feature),
                request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                text=text, aux=aux, usage=usage, finish_reason=reason, events=events)


async def run(args):
    args.result_dir.mkdir(parents=True, exist_ok=True)
    images = fixtures(args.result_dir / 'fixtures')
    if args.include_video and not args.video_file:
        args.video_file = args.result_dir / 'fixtures' / 'color_change.mp4'
        make_video_fixture(args.video_file)
    fixed = cases(images)
    (args.result_dir / 'requests.json').write_text(json.dumps(fixed, indent=2))
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), trust_env=False,
                                     connector=aiohttp.TCPConnector(force_close=True)) as session:
        for case in fixed:
            for phase, cache in [('no_cache', False), ('cache_prime', True), ('cache_hit', True)]:
                row = await request(session, args.base_url, case, cache, phase)
                rows.append(row)
                (args.result_dir / 'result.json').write_text(json.dumps(rows, indent=2))
                print(case['name'], phase, row['correct'], row['feature_ok'], row['aux'].get('reuse_len'), repr(row['text'][:160]), flush=True)
        # Mixed text/image scheduling, including non-square MRoPE and padded batches.
        mixed = [fixed[-1], {**fixed[5], 'name': 'tall_long', 'long': True,
                 'messages': [{'role': 'user', 'content': [
                     {'type': 'image_url', 'image_url': {'url': images['tall']}},
                     {'type': 'text', 'text': 'Describe the colors, shapes and vertical arrangement in detail in at least 80 English words.'}]}],
                 'expected': ['red', 'blue', 'above|top']},
                 {'name': 'text_mixed', 'images': 0, 'long': True, 'expected': ['four|4'],
                  'messages': [{'role': 'user', 'content': 'Explain in at least 80 English words why two plus two equals four, using counting examples.'}]}]
        rows.extend(await asyncio.gather(*(request(session, args.base_url, c, True, 'mixed_c3') for c in mixed)))
        if args.video_file:
            for case in video_cases(args.video_file):
                for phase, cache in [('no_cache', False), ('cache_prime', True), ('cache_hit', True)]:
                    rows.append(await request(session, args.base_url, case, cache, phase))
    (args.result_dir / 'result.json').write_text(json.dumps(rows, indent=2))
    summary = {'requests': len(rows), 'correct': sum(r['correct'] for r in rows), 'feature_ok': sum(r['feature_ok'] for r in rows)}
    if args.reference:
        reference = json.loads(args.reference.read_text())
        assert [(r['name'], r['phase'], r['request_sha256']) for r in reference] == [(r['name'], r['phase'], r['request_sha256']) for r in rows]
        summary['exact_text_matches'] = sum(a['text'] == b['text'] for a, b in zip(reference, rows))
    (args.result_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)
    assert len(rows) == (42 if args.video_file else 36) and all(r['correct'] and r['feature_ok'] for r in rows), 'Multimodal regression failed; inspect per-request result.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:29700')
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--video-file', type=Path, help='Optional fixed red-to-blue square MP4 fixture')
    parser.add_argument('--include-video', action='store_true', help='Generate the fixed MP4 fixture using PyAV and add six video cases')
    asyncio.run(run(parser.parse_args()))
