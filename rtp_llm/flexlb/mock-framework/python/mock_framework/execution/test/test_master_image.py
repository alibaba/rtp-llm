"""Check the Mock overlay against the layout used by the Whale Master archive."""

import io
import subprocess
import sys
import tarfile
import zipfile

from mock_framework.execution.test.helpers import FRAMEWORK


def test_overlay_selects_the_provider_beside_the_vipserver_client(tmp_path):
    master = tmp_path / "master.tgz"
    mock_jar = tmp_path / "mock-master.jar"
    output = tmp_path / "mock.tgz"
    package = "org/flexlb"
    with zipfile.ZipFile(mock_jar, "w") as archive:
        archive.writestr(f"{package}/WhaleMockMasterMain.class", b"entry")
        archive.writestr(f"{package}/MockDpServiceDiscoveryProvider.class", b"wrapper")
    provider = io.BytesIO()
    with zipfile.ZipFile(provider, "w") as archive:
        archive.writestr(
            f"{package}/discovery/VipServerDiscovery.class",
            b"org/flexlb/discovery/ServiceDiscoveryProvider",
        )
    manifest = "FlexLB/META-INF/MANIFEST.MF"
    with tarfile.open(master, "w:gz") as archive:
        payloads = {
            manifest: b"Manifest-Version: 1.0\r\nStart-Class: org.flexlb.Application\r\n",
            "FlexLB/BOOT-INF/lib/vipserver-1.0.0-SNAPSHOT.jar": provider.getvalue(),
            "FlexLB/BOOT-INF/lib/vipserver-client-1.0.0.jar": b"client",
        }
        for path, payload in payloads.items():
            entry = tarfile.TarInfo(path)
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))

    subprocess.run(
        [
            sys.executable,
            FRAMEWORK / "scripts/package-whale-master.py",
            master,
            mock_jar,
            output,
        ],
        check=True,
    )
    with tarfile.open(output, "r:gz") as archive:
        assert (
            b"Start-Class: org.flexlb.WhaleMockMasterMain\r\n"
            in archive.extractfile(manifest).read()
        )
        assert (
            archive.extractfile(
                f"FlexLB/BOOT-INF/classes/{package}/WhaleMockMasterMain.class"
            ).read()
            == b"entry"
        )
