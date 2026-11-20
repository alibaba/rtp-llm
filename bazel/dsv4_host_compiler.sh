#!/bin/bash
set -euo pipefail
export LC_ALL=C
export PATH=/usr/bin:/bin
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH

describe() {
    local compiler="$1"
    local version target tool path headers
    local prefix="-B${compiler%/*}/"
    if [[ "$compiler" != /* || ! -x "$compiler" ]]; then
        echo "The private SM120 binding requires GCC 12. Set RTP_DSV4_HOST_CXX to its absolute path." >&2
        return 1
    fi
    version="$("$compiler" -dumpfullversion)"
    target="$("$compiler" -dumpmachine)"
    if [[ "$version" != 12.* || "$target" != x86_64*-linux* ]]; then
        echo "The private SM120 binding requires GCC 12 for x86_64 Linux, got $version for $target." >&2
        return 1
    fi
    printf '%s\n' "$compiler" "$version" "$target"
    for tool in cc1plus as ld collect2; do
        path="$("$compiler" "$prefix" "-print-prog-name=$tool")"
        if [[ "$path" != /* ]]; then path="$(command -v "$path")"; fi
        sha256sum "$path"
    done
    sha256sum "$compiler"
    for tool in libstdc++.so libstdc++.so.6 libgcc.a libgcc_s.so.1 libc.so libc.so.6 libm.so crtbeginS.o crtendS.o; do
        path="$("$compiler" "$prefix" "-print-file-name=$tool")"
        if [[ ! -f "$path" ]]; then
            echo "Missing GCC 12 link input: $path" >&2
            return 1
        fi
        sha256sum "$path"
    done
    # These implicit headers come from the execution image. Compare their
    # contents as well as the front end and binutils before using the image.
    headers="$("$compiler" "$prefix" -E -v -x c++ /dev/null -o /dev/null 2>&1         | sed -n '/^#include <\.\.\.> search starts here:$/,/^End of search list\.$/s/^ //p')"
    if [[ -z "$headers" ]]; then
        echo "Could not resolve the GCC 12 implicit include paths." >&2
        return 1
    fi
    while IFS= read -r path; do
        printf '%s\n' "$path"
        find -L "$path" -type f -print0 | sort -z | xargs -0 -r sha256sum | sha256sum
    done <<< "$headers"
}

if [[ "$1" == --describe ]]; then
    describe "$2"
    exit
fi

identity="$1"
shift
IFS= read -r compiler < "$identity"
actual="$(describe "$compiler")"
if [[ "$actual" != "$(cat "$identity")" ]]; then
    echo "The execution image's GCC 12, binutils, or sysroot differs from the configured SM120 toolchain." >&2
    exit 1
fi
exec "$compiler" "-B${compiler%/*}/" "$@"
