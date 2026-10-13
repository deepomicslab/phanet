"""Restartable data preparation, with completion markers written last."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import os
import shutil

from phanet.common.io import digest_file, digest_object, read_json, write_json


def source_signature(paths):
    """Identify input archives by path, size and modification time."""
    return {str(Path(p).resolve()): {"size": Path(p).stat().st_size,
                                   "mtime_ns": Path(p).stat().st_mtime_ns} for p in sorted(paths)}


def _resume_identity_matches(previous, current):
    """Resume completed stages by checksum; rebuild incomplete stages after code updates."""
    if previous == current:
        return True
    previous = dict(previous); current = dict(current)
    previous_config = dict(previous.get("config", {})); current_config = dict(current.get("config", {}))
    previous_config.pop("implementation", None); current_config.pop("implementation", None)
    previous["config"] = previous_config; current["config"] = current_config
    return previous == current


@contextmanager
def preparation_stage(output, stage, config, *, resume=False):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Lock outside the output so an existing non-stage directory stays untouched.
    with (output.parent / (output.name + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another preparation process owns {output}") from exc
        identity = {"stage": stage, "config": config}
        state_path = output / ".preparation.json"
        manifest_path = output / "manifest.json"
        if output.exists():
            if not resume or not state_path.is_file():
                raise FileExistsError(f"Use a fresh output or --resume for a matching stage: {output}")
            state = read_json(state_path)
            exact_identity = state["identity"] == identity
            if not _resume_identity_matches(state["identity"], identity):
                raise ValueError("Resume input/configuration changed; choose a fresh output")
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                if manifest.get("status") != "complete":
                    raise ValueError("Invalid completion manifest")
                for name, digest in manifest["outputs"].items():
                    path = (output / name).resolve()
                    if not path.is_relative_to(output) or digest_file(path) != digest:
                        raise ValueError(f"Prepared output integrity failure: {name}")
                yield None
                return
            if state.get("status") == "failed":
                # Rebuild temporary files for the incomplete stage.
                    for child in output.iterdir():
                        if child.name not in (".preparation.json", ".cache"):
                            shutil.rmtree(child) if child.is_dir() else child.unlink()
        else:
            output.mkdir()
        write_json(state_path, {"identity": identity, "status": "running", "pid": os.getpid()})
        try:
            yield output
            # Callers write their manifest after closing all output streams.
            if not manifest_path.is_file():
                raise RuntimeError("Preparation finished without a completion manifest")
            write_json(state_path, {"identity": identity, "status": "complete"})
        except BaseException as exc:
            write_json(state_path, {"identity": identity, "status": "failed", "error": str(exc)})
            raise


def complete_preparation(output, summary, *, paths=None):
    output = Path(output)
    files = paths if paths is not None else [p for p in output.iterdir() if p.is_file() and not p.name.startswith(".")]
    manifest = {**summary, "status": "complete", "schema_version": 1,
                "outputs": {str(p.relative_to(output)): digest_file(p) for p in sorted(files)
                            if p.name != "manifest.json"}}
    write_json(output / "manifest.json", manifest)
    return manifest


def segment_starts(n, size=30, overlap=0):
    """Native-order segments; overlap>0 shifts the final full window to the end."""
    if n < 0 or size < 1 or not 0 <= overlap < size:
        raise ValueError("Invalid segment dimensions")
    if not n:
        return ()
    if not overlap:
        return range(0, n, size)
    last = max(0, n - size)
    starts = list(range(0, last + 1, size - overlap))
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)
