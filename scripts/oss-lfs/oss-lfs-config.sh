#!/bin/bash
# Resolve OSS credentials for oss-lfs.
# Priority: env vars > ~/.oss-lfs-config > error

OSS_LFS_BUCKET="${OSS_LFS_BUCKET:-rtp-opensource}"
OSS_LFS_ENDPOINT="${OSS_LFS_ENDPOINT:-oss-cn-hangzhou.aliyuncs.com}"
OSS_LFS_REGION="${OSS_LFS_REGION:-cn-hangzhou}"

if [ -z "$OSS_LFS_ACCESS_KEY_ID" ] || [ -z "$OSS_LFS_ACCESS_KEY_SECRET" ]; then
    if [ -f "$HOME/.oss-lfs-config" ]; then
        source "$HOME/.oss-lfs-config"
    fi
fi

if [ -z "$OSS_LFS_ACCESS_KEY_ID" ] || [ -z "$OSS_LFS_ACCESS_KEY_SECRET" ]; then
    echo "oss-lfs: credentials not configured. Set OSS_LFS_ACCESS_KEY_ID and OSS_LFS_ACCESS_KEY_SECRET env vars, or create ~/.oss-lfs-config" >&2
    return 1 2>/dev/null || exit 1
fi
