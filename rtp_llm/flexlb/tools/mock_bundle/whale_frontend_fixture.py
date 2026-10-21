"""Start release frontend or CPU ViT with real DSV4.1 checkpoint metadata.

The historical filename is the CI packaging contract. No synthetic tokenizer,
model config, native symbol replacement or model weights are generated here.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys


def prepare_assets(source, destination):
    if not source.startswith('oss://'):
        path = Path(source)
        if not (path / 'config.json').is_file():
            raise ValueError('model path lacks config.json')
        return path
    import base64
    import email.utils
    import hmac
    import urllib.error
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET
    url = urllib.parse.urlsplit(source)
    query = dict(urllib.parse.parse_qsl(url.query))
    # Keep the deployment's regional endpoint. Credentials never enter artifacts/logs.
    endpoint = query['OSS_ENDPOINT']
    prefix = url.path.strip('/')
    destination.mkdir(parents=True, exist_ok=True)
    def get(key, listing=False):
        date = email.utils.formatdate(usegmt=True)
        resource = '/' + url.netloc + '/' + ('' if listing else key)
        signature = base64.b64encode(hmac.new(query['OSS_ACCESS_KEY'].encode(),
            ('GET\n\n\n' + date + '\n' + resource).encode(), hashlib.sha1).digest()).decode()
        path = '/?' + urllib.parse.urlencode({'prefix': key, 'max-keys': 1000}) if listing else '/' + urllib.parse.quote(key, safe='/')
        request = urllib.request.Request('https://' + url.netloc + '.' + endpoint + path,
            headers={'Date': date, 'Authorization': 'OSS ' + query['OSS_ACCESS_ID'] + ':' + signature})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(64 * 1024 * 1024 + 1)
                if len(data) > 64 * 1024 * 1024:
                    raise ValueError('model metadata file exceeds byte budget')
                return data
        except urllib.error.HTTPError as error:
            if error.code == 404 and not listing:
                return None
            raise RuntimeError('model metadata fetch HTTP ' + str(error.code)) from None
    files = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'generation_config.json', 'tokenizer.model', 'vocab.json', 'merges.txt', 'chat_template.jinja',
             'processor_config.json', 'preprocessor_config.json']
    listing = ET.fromstring(get(prefix + '/encoding/', True))
    if listing.findtext('IsTruncated', 'false') == 'true':
        raise ValueError('encoding listing truncated')
    for item in listing.findall('Contents'):
        key, size = item.findtext('Key'), int(item.findtext('Size', '0'))
        if key and 0 < size <= 64 * 1024 * 1024:
            files.append(key[len(prefix) + 1:])
    def fetch(name):
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('invalid metadata path')
        data = get(prefix + '/' + name)
        if data is not None:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, files))
    if not (destination / 'config.json').is_file():
        raise ValueError('checkpoint config missing')
    manifest = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in destination.rglob('*') if p.is_file() and p.name != 'asset-sha256.json'}
    (destination / 'asset-sha256.json').write_text(json.dumps(manifest, sort_keys=True, indent=2))
    return destination


def main():
    if os.environ.get('RTP_LLM_MOCK_FRONTEND') != '1':
        raise ValueError('release test deployment requires RTP_LLM_MOCK_FRONTEND=1')
    source = Path(os.environ.get('MOCK_FRONTEND_SOURCE_ROOT', str(Path(__file__).parents[4])))
    sys.path.insert(0, str(source))
    runtime = Path(os.environ.get('HIPPO_PROC_WORKDIR', '/tmp')) / 'legacy-vit-assets'
    model = prepare_assets(os.environ['CHECKPOINT_PATH'], runtime)
    tokenizer = os.environ.get('TOKENIZER_PATH', os.environ['CHECKPOINT_PATH'])
    tokenizer = model if tokenizer == os.environ['CHECKPOINT_PATH'] else prepare_assets(tokenizer, runtime / 'tokenizer')
    os.environ.update(CHECKPOINT_PATH=str(model), TOKENIZER_PATH=str(tokenizer))
    if os.environ.get('MOCK_BUNDLE_ROLE') == 'vit':
        from rtp_llm.server.mock_vit_engine import main as vit_main
        sys.argv = [sys.argv[0], '--port', os.environ.get('MOCK_VIT_PORT', '22161'),
                    '--health-port', os.environ.get('START_PORT', '22160'), '--prepared-v41',
                    '--model-path', str(model), '--concurrency', os.environ.get('VIT_MAX_CONCURRENT_REQUESTS', '32'),
                    '--cache-bytes', os.environ.get('MOCK_VIT_CACHE_BYTES', '0')]
        vit_main()
        return
    # Fail fast on a mismatched native library. No Python substitute for native hashing.
    from rtp_llm.ops import DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED, cpp_get_block_cache_keys
    assert DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED == 0x4453563431525031
    assert cpp_get_block_cache_keys([[1, 2]], 123) == [-8366447758695274051]
    from rtp_llm.server.server_args.server_args import setup_args
    from rtp_llm.frontend.frontend_app import FrontendApp
    from rtp_llm.config.log_config import setup_logging
    from rtp_llm.utils.concurrency_controller import ConcurrencyController, set_global_controller
    setup_logging()
    cfg = setup_args([])
    set_global_controller(ConcurrencyController(int(os.environ.get('CONCURRENCY_LIMIT', '128'))))
    FrontendApp(cfg, separated_frontend=True).start()


if __name__ == '__main__':
    main()
