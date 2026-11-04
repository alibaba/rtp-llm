# Captured pending reservations

`agent-316-reservations.txt` contains the unmodified reservation IDs and hash
vectors of one agent view read from Master PID 446651 at Unix time
1792622971.6898246 (2026-10-22). Agent: 215.69.233.18:27414.
Exact loaded Master core SHA256:
`c7a8165f330fd065d135a3883a3c61f8c2f9fee20a6e4f0b2b4e542f3dbafe1a`.

Capture validates unchanged selected state/revision around vector reads and
recomputes the full reservation checksum. Two changing attempts were rejected
before the accepted capture. Independent local checksum, vector uniqueness and
owner cardinalities agree. Original decoded capture SHA256:
`c9d3afaddbd7a3ae13d05c300d2c240a98858caa4849be8bbbf403a20a092980`.

First line: batch count, checksum, full unique owners, tail unique owners.
Each subsequent line: batch ID, schedule batch ID, full hash count, full hashes,
tail hash count, tail hashes. Values are decimal integers to retain uint64
precision. Trailing whitespace is normalized; all numeric values are unchanged.
There are 316 batches, 215311 full references and 177564 distinct
full hashes. No prompt text or tokens are included.

The regression chooses hit and miss probes from this real structure. Probes
are synthetic; this is not the historical 45-second request sequence and does
not establish that window's first trigger.

`master-orphan-28.txt` captures batch 7711266218097843350 from the same
Master core at Unix time 1792628038.1085298 (2026-10-22). The collector is
33.68.218.88:24854. It contains 28 requests and 6011 block hashes, with 27
FAILED requests and one unacknowledged WAITING request, ID
7286868037547545303. The batch was reserved at 07:48:04 and remained in
read-only observations at 08:01, 08:07 and 08:13. Concurrent capture validates
batch headers; it is not an atomic process snapshot or a historical event tape.

Header: batch ID, schedule time in microseconds, predicted batch latency,
policy block size, linear step, hybrid-tail flag, request count. Request lines:
request ID, total tokens, block size, predicted cache-hit tokens, predicted
latency, captured state (0 WAITING / 2 FAILED), ACK flag, hash count, hashes.
All IDs and hashes remain decimal integers. No prompt text or token IDs are
included. File SHA256:
`8f214f07e0c8511aa624231d9c7fb5baa5e9468a3d447f3558d196344b5dcae6`.

The regression reconstructs reservation creation and the captured 27 failures,
then explicitly abandons the last request. It checks that the actual request
and hash structure can be cleaned when the lifecycle notification is delivered.
The test uses an isolated empty static cache; it does not reproduce historical
cache residency, network timing, response acceptance, or the 45-second sequence.
