"""Download, verify and install eggNOG database resources."""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import time
import urllib.request

from phanet.common.io import digest_file, write_json

VERSION = "5.0.2"
BASE_URL = "http://eggnog5.embl.de/download/emapperdb-5.0.2/"
# SHA-256 checksums for the supported upstream database files.
FILES = {
    "eggnog.db": (41370988544, "4d6857c129423889edcd607995b28d97944787a588224f31dd7f745fe9ef1e4f"),
    "eggnog.taxa.db": (278003712, "84d56ba1cbb091cc4c4064ba0f8235020654ed50fd6d617f829c4e45551faffd"),
    "eggnog.taxa.db.traverse.pkl": (6628719, "de7bb2b35489c9e5e1f2522575c8107ebd3d7117f8910dc6745aab68ac2d2173"),
    "eggnog_proteins.dmnd": (9285439161, "f4b63db0313021e4a2ca1d47a6b3f7862a3c8c2016abf3b51ef5865e653642a7"),
}
ARCHIVES = (
    ("eggnog.db.gz", 6776977123, "20fd3e57d96ab19f2094131e859cb0f2e634e10f594c01ef34335076317a8e0a", ("eggnog.db",)),
    ("eggnog.taxa.tar.gz", 72797584, "ead5f9f1cb42a51a990c3980fcb272a2f6c07b41c75654739e799d96a190a7f3", ("eggnog.taxa.db", "eggnog.taxa.db.traverse.pkl")),
    ("eggnog_proteins.dmnd.gz", 5208806170, "c74fe142c6d0f1c749a086f754f1658874c0a52d0927626b9d263098c136b320", ("eggnog_proteins.dmnd",)),
)
CHUNK = 1024 * 1024


def _regular(path):
    if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink != 1)):
        raise ValueError(f"Expected an unshared regular file: {path}")


def _verify(path, size, sha):
    _regular(path)
    if path.stat().st_size != size or digest_file(path) != sha:
        raise ValueError(f"Integrity check failed: {path}; preserve it for diagnosis and use a new database directory")


@contextmanager
def _lock(root):
    path = root / ".setup.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Database setup is locked: {path}. After a hard interruption, confirm the writer stopped before removing this lock.") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def _download(url, path, size, sha, progress):
    """Keep partial bytes on disconnect; append only a validated HTTP range."""
    _regular(path)
    if path.exists():
        _verify(path, size, sha)
        return
    part = path.with_name(path.name + ".part")
    _regular(part)
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError(f"Oversized partial download: {part}; use a new database directory")
    if offset < size:
        headers = {"User-Agent": "PhaNet/0.1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unexpected HTTP content encoding")
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or tuple(map(int, match.groups())) != (offset, size - 1, size):
                        raise ValueError("Invalid HTTP range response; partial file left unchanged")
                elif response.status == 200:
                    offset = 0  # Server ignored Range: restart, never append a full response.
                else:
                    raise ValueError(f"Unexpected HTTP status: {response.status}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != size - offset:
                    raise ValueError("Unexpected HTTP download size; partial file left unchanged")
                progress(f"Downloading {path.name}: {offset:,}/{size:,} bytes")
                last = time.monotonic()
                with part.open("ab" if offset else "wb") as stream:
                    while block := response.read(CHUNK):
                        if offset + len(block) > size:
                            raise ValueError(f"Download exceeds expected size: {url}")
                        stream.write(block)
                        offset += len(block)
                        if time.monotonic() - last >= 10:
                            progress(f"Downloading {path.name}: {offset:,}/{size:,} bytes")
                            last = time.monotonic()
        except (OSError, http.client.HTTPException) as exc:
            raise RuntimeError(f"Download interrupted: {url}; rerun the same setup command to resume ({exc})") from exc
        if offset != size:
            raise RuntimeError(f"Incomplete download: {part}; rerun the same setup command to resume")
    progress(f"Checking archive SHA-256: {path.name}")
    _verify(part, size, sha)
    part.replace(path)


def _unpack(archive, members, stage):
    def copy_checked(stream, name):
        expected, sha = FILES[name]
        count, digest = 0, hashlib.sha256()
        with (stage / name).open("xb") as out:
            while block := stream.read(CHUNK):
                count += len(block)
                if count > expected:
                    raise ValueError(f"Expanded file exceeds expected size: {name}")
                digest.update(block)
                out.write(block)
        if count != expected or digest.hexdigest() != sha:
            raise ValueError(f"Expanded file integrity check failed: {name}")

    if archive.name.endswith(".tar.gz"):
        seen = set()
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                # Upstream uses ./name; accept that spelling without general path normalization.
                name = member.name[2:] if member.name.startswith("./") else member.name
                if name not in members or name in seen or not member.isfile():
                    raise ValueError(f"Unexpected or unsafe archive member: {member.name}")
                if member.size != FILES[name][0]:
                    raise ValueError(f"Unexpected archive member size: {member.name}")
                seen.add(name)
                with tar.extractfile(member) as stream:
                    copy_checked(stream, name)
        if seen != set(members):
            raise ValueError("Missing database archive members")
    else:
        with gzip.open(archive, "rb") as stream:
            copy_checked(stream, members[0])


def install_eggnog(directory, *, progress=print):
    """Install or verify DB 5.0.2. Existing files are checked and never replaced."""
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root):
        cache = root / ".downloads"
        if cache.is_symlink():
            raise ValueError(f"Download cache must not be a symlink: {cache}")
        cache.mkdir(exist_ok=True)
        receipt = root / "phanet-eggnog.json"
        _regular(receipt)
        present = set()
        for name, (size, sha) in FILES.items():
            p = root / name
            _regular(p)
            if p.exists():
                progress(f"Checking installed SHA-256: {name}")
                _verify(p, size, sha)
                present.add(name)
        needed = [a for a in ARCHIVES if not set(a[3]) <= present]
        # Cache archives are retained for interrupted extraction or later repair.
        required = 1024**3 if needed else 0
        for name, size, sha, members in needed:
            archive = cache / name
            part = cache / (name + ".part")
            _regular(archive)
            _regular(part)
            cached = archive.stat().st_size if archive.exists() else (part.stat().st_size if part.exists() else 0)
            required += max(0, size - cached) + sum(FILES[n][0] for n in members)
        if shutil.disk_usage(root).free < required:
            raise OSError(f"Insufficient free space: need about {required / 10**9:.1f} GB in {root}")
        try:
            for name, size, sha, members in needed:
                archive = cache / name
                _download(BASE_URL + name, archive, size, sha, progress)
                progress(f"Extracting and checking {name}")
                with tempfile.TemporaryDirectory(prefix=".extract-", dir=root) as temp:
                    stage = Path(temp)
                    _unpack(archive, members, stage)
                    for member in members:
                        target = root / member
                        if member in present:
                            continue
                        # Refuse concurrent changes even outside this installer's lock.
                        os.link(stage / member, target)
                        (stage / member).unlink()
                        present.add(member)
        except (tarfile.TarError, EOFError) as exc:
            raise ValueError(f"Database extraction failed: {exc}") from exc
        result = {"schema_version": 1, "database": "eggNOG", "version": VERSION,
                  "files": {n: {"size": s, "sha256": h} for n, (s, h) in FILES.items()},
                  "sources": [{"url": BASE_URL + n, "size": s, "sha256": h} for n, s, h, _ in ARCHIVES]}
        # Stable content allows prepare --resume after a repeated setup verification.
        if not receipt.exists():
            write_json(receipt, result)
        elif json.loads(receipt.read_text()) != result:
            raise ValueError(f"Existing installation receipt differs: {receipt}")
        progress(f"eggNOG {VERSION} ready: {root}")
        return result
