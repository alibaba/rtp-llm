"""Small, deadline-bound HTTP transport. Mutations are never retried."""
import asyncio
import json
from urllib.parse import urlsplit


class ExecutionError(RuntimeError):
    pass


async def post(url, payload, *, limit=16 * 1024 * 1024):
    address = urlsplit(url)
    if address.scheme != 'http' or not address.hostname or address.username or address.query or address.fragment:
        raise ExecutionError('Component endpoint must be an HTTP URL without credentials/query/fragment')
    reader, writer = await asyncio.open_connection(address.hostname, address.port or 80, limit=65536)
    try:
        body = json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()
        path = (address.path.rstrip('/') + '/invoke')
        writer.write((f'POST {path} HTTP/1.1\r\nHost: {address.netloc}\r\n'
                      f'Content-Type: application/json\r\nContent-Length: {len(body)}\r\n'
                      'Connection: close\r\n\r\n').encode() + body)
        await writer.drain()
        header = await reader.readuntil(b'\r\n\r\n')
        lines = header.decode('ascii').split('\r\n')
        status = int(lines[0].split()[1])
        headers = {}
        for line in lines[1:]:
            if line:
                key, value = line.split(':', 1)
                key = key.lower()
                if key in headers:
                    raise ExecutionError('Duplicate response header')
                headers[key] = value.strip()
        if 'transfer-encoding' in headers:
            raise ExecutionError('Component response must use Content-Length')
        length = int(headers['content-length'])
        if not 0 <= length <= limit:
            raise ExecutionError('Component response is too large')
        raw = await reader.readexactly(length)
        response = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if status != 200:
            raise ExecutionError(f'Component HTTP {status}: {response}')
        if not isinstance(response, dict) or set(response) != {'result'}:
            raise ExecutionError('Invalid component response envelope')
        return response['result']
    except (ValueError, KeyError, UnicodeError, asyncio.IncompleteReadError, asyncio.LimitOverrunError) as error:
        raise ExecutionError(f'Invalid HTTP response: {error}') from error
    finally:
        writer.close()
        try:
            async with asyncio.timeout(.5):
                await writer.wait_closed()
        except (OSError, TimeoutError):
            writer.transport.abort()
