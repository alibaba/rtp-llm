#!/usr/bin/env bash

restore_image_library_path() {
    local image_library_path
    [ -r "$1" ] || return 0
    image_library_path=$(cat "$1") || return
    [ -n "$image_library_path" ] || return 0

    # Deployment paths retain priority; image dependencies remain discoverable.
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${image_library_path}"
}
