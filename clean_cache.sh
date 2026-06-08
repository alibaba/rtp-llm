#!/bin/bash
# /data1/hengcang.wyd/RTP-LLM/github-opensource/clean_cache.sh
set -e

CKPT=${CKPT:-/data1/hengcang.wyd/models/qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796}

PYTHON=""
for p in /opt/conda310/bin/python3 python3 python /usr/bin/python3 /usr/bin/python; do
    if command -v "$p" >/dev/null 2>&1; then PYTHON=$(command -v "$p"); break; fi
done
if [ -z "$PYTHON" ]; then
    echo "no python interpreter found (tried conda310, python3, python)"; exit 1
fi
echo "==> using $PYTHON"

# NOTE: across containers/sessions the same user can have different UIDs but a
# stable group. Match by group to catch all of "your" files/processes safely.
CLEAN_GROUP="${CLEAN_GROUP:-hengcang.wyd}"
echo "==> using group=$CLEAN_GROUP for cleanup (override with CLEAN_GROUP=...)"

echo "==> kill orphan RTP-LLM processes by group (covers multi-UID sessions)"
pkill -G "$CLEAN_GROUP" -f "rtp_llm_rank|backend_server|start_server" 2>/dev/null || true
sleep 2
pkill -9 -G "$CLEAN_GROUP" -f "rtp_llm_rank|backend_server|start_server" 2>/dev/null || true
sleep 1
pkill -9 -u "$USER" -f "rtp_llm_rank|frontend_server|start_server" 2>/dev/null || true

echo "==> remove fastsafetensors shm by group"
SHM_BEFORE=$(find /dev/shm -name "P*F-preallocate" -group "$CLEAN_GROUP" 2>/dev/null | wc -l)
find /dev/shm -name "P*F-preallocate" -group "$CLEAN_GROUP" -delete 2>/dev/null || true
SHM_AFTER=$(find /dev/shm -name "P*F-preallocate" -group "$CLEAN_GROUP" 2>/dev/null | wc -l)
echo "  shm files (group=$CLEAN_GROUP): $SHM_BEFORE -> $SHM_AFTER"

echo "==> evict page cache for $CKPT (recursive, follows symlinks)"
"$PYTHON" -c "
import os, glob
total = 0
files = glob.glob('$CKPT/**/*', recursive=True) + glob.glob('$CKPT/*')
seen = set()
for f in files:
    rp = os.path.realpath(f)
    if rp in seen or not os.path.isfile(rp):
        continue
    seen.add(rp)
    size = os.path.getsize(rp)
    fd = os.open(rp, os.O_RDONLY)
    os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    total += size
print(f'  evicted: {total/1024**3:.2f} GiB across {len(seen)} files')
"

echo "==> verify cache state (Cached value in /proc/meminfo)"
grep -E '^(MemFree|Cached|Buffers):' /proc/meminfo

echo "==> done, ready for cold-start test"
