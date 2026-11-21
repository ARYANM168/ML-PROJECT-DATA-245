"""Provenance for raw downloads: SHA-256 checksums, read-only files, and a manifest."""

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path


def sha256_of(path, chunk_bytes=1 << 20):
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_read_only(path):
    """Remove write permission for owner, group, and others."""
    current = stat.S_IMODE(os.stat(path).st_mode)
    os.chmod(path, current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def record_download(path, source_url, pudl_version=None, manifest_path=None):
    """Checksum a downloaded raw file, make it read-only, and append to a JSON manifest.

    The manifest is a list of records. Each record holds filename, sha256,
    source URL, UTC download timestamp, and PUDL version (if used).
    """
    path = Path(path)
    record = {
        "file": path.name,
        "sha256": sha256_of(path),
        "bytes": path.stat().st_size,
        "source_url": source_url,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pudl_version": pudl_version,
    }
    make_read_only(path)
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        records = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        records = [r for r in records if r["file"] != record["file"]]
        records.append(record)
        manifest_path.write_text(json.dumps(records, indent=2))
    return record


def verify_manifest(manifest_path, raw_dir):
    """Return a list of (file, problem) for any file whose checksum no longer matches."""
    problems = []
    for rec in json.loads(Path(manifest_path).read_text()):
        target = next(Path(raw_dir).rglob(rec["file"]), None)  # searches subfolders too
        if target is None:
            problems.append((rec["file"], "missing"))
        elif sha256_of(target) != rec["sha256"]:
            problems.append((rec["file"], "checksum mismatch"))
    return problems
