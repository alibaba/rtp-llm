"""Install site for the PPU wheel hub.

The PPU wheel graph ships from the platform distribution, so the open-source
tree has no @pip_ppu_torch hub to install. Internal builds override @rtp_deps
with a copy of this file that installs the hub declared by their pip.bzl;
keeping one load site in WORKSPACE avoids branching on the distribution.
"""

def pip_ppu_install_deps():
    """No-op: the open-source tree has no PPU wheel hub."""
