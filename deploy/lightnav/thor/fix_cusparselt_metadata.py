"""Repair the nonstandard sbsa tag in NVIDIA's 0.8.0 AArch64 wheel metadata.

The downloaded filename says manylinux2014_aarch64, but its internal WHEEL
file says manylinux2014_sbsa. Verify the ELF before correcting only that tag.
"""

import base64
import csv
import hashlib
from importlib.metadata import distribution

dist = distribution("nvidia-cusparselt-cu13")
assert dist.version == "0.8.0", dist.version
wheel_entry = next(p for p in dist.files if str(p).endswith(".dist-info/WHEEL"))
wheel = dist.locate_file(wheel_entry)
metadata = wheel.read_text()
if "Tag: py3-none-manylinux2014_sbsa" in metadata:
    libraries = [p for p in dist.files if ".so" in str(p)]
    assert libraries, "No cuSPARSELt libraries found"
    for lib in libraries:
        with dist.locate_file(lib).open("rb") as f:
            header = f.read(20)
        assert header[:6] == b"\x7fELF\x02\x01", lib
        assert int.from_bytes(header[18:20], "little") == 183, lib  # EM_AARCH64
    wheel.write_text(metadata.replace("manylinux2014_sbsa", "manylinux2014_aarch64"))
    record = wheel.with_name("RECORD")
    with record.open(newline="") as f:
        rows = list(csv.reader(f))
    digest = base64.urlsafe_b64encode(hashlib.sha256(wheel.read_bytes()).digest()).rstrip(b"=")
    for row in rows:
        if row[0] == str(wheel_entry):
            row[1:] = ["sha256=" + digest.decode(), str(wheel.stat().st_size)]
    with record.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("Corrected cuSPARSELt WHEEL tag after verifying AArch64 ELF binaries")
