"""Install annotation tools into a separate Conda environment."""
from pathlib import Path
import os
import shutil
import subprocess

from phanet.annotation import _probe
from phanet.common.io import read_json, write_json
from phanet.resources import _lock

SPECS = ("python=3.10", "prodigal=2.6.3", "eggnog-mapper=2.1.13", "diamond=2.0.11")


def tool_commands(prefix):
    prefix = Path(prefix).expanduser().resolve()
    return {"prodigal": str(prefix / "bin/prodigal"), "emapper": [str(prefix / "bin/python"), "-I", str(prefix / "bin/emapper.py")]}


def install_tools(prefix, *, manager="conda", progress=print):
    """Create an isolated tool environment or reuse a matching installation."""
    root = Path(prefix).expanduser().absolute()
    if root.is_symlink():
        raise ValueError("Tool install directory must not be a symlink")
    root.parent.mkdir(parents=True, exist_ok=True)
    # A private control directory keeps logs outside the Conda prefix.
    control = root.with_name(root.name + ".phanet-install")
    if control.is_symlink():
        raise ValueError("Tool install control directory must not be a symlink")
    control.mkdir(exist_ok=True)
    with _lock(control):
        receipt = control / "COMPLETE.json"
        if root.exists():
            if not receipt.is_file():
                raise FileExistsError(f"Existing or incomplete environment: {root}; choose a new --tools-prefix")
            saved = read_json(receipt)
            if saved.get("specs") != list(SPECS):
                raise ValueError("Tool environment specification changed; choose a new --tools-prefix")
        else:
            executable = shutil.which(manager)
            if not executable:
                raise FileNotFoundError("Install Conda or Mamba first, or provide --manager /path/to/conda")
            command = [executable, "create", "--yes", "--prefix", str(root), "--override-channels",
                       "--channel", "conda-forge", "--channel", "bioconda", "--strict-channel-priority", *SPECS]
            write_json(control / "command.json", {"argv": command})
            progress(f"Installing annotation tools in {root}; log: {control / 'install.log'}")
            with (control / "install.log").open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "CONDA_REPORT_ERRORS": "false"})
            if result.returncode:
                raise RuntimeError(f"Tool installation failed; see {control / 'install.log'}; retry with a new --tools-prefix")
        commands = tool_commands(root)
        versions = {"prodigal": _probe([commands["prodigal"], "-v"]),
                    "diamond": _probe([str(root / "bin/diamond"), "version"])}
        if "2.6.3" not in versions["prodigal"] or "2.0.11" not in versions["diamond"]:
            raise ValueError("Installed annotation tool versions differ from pinned versions")
        records = [read_json(p) for p in (root / "conda-meta").glob("*.json")]
        if not any(p.get("name") == "eggnog-mapper" and p.get("version") == "2.1.13" for p in records):
            raise ValueError("eggNOG-mapper 2.1.13 installation not found")
        if not Path(commands["emapper"][-1]).is_file():
            raise FileNotFoundError(commands["emapper"][-1])
        if receipt.exists():
            if saved.get("versions") != versions:
                raise ValueError("Tool environment changed since setup")
        else:
            write_json(receipt, {"specs": list(SPECS), "versions": versions})
        progress("Annotation tools ready")
        return commands
