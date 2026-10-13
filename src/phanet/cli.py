"""Command-line interfaces for annotation, resource setup and downstream analysis."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

from phanet import __version__
from phanet.annotation import PrepareConfig, preflight, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phanet", description="PhaNet genome annotation")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    from phanet.downstream.evidence_cli import add_parsers
    add_parsers(sub)
    from phanet.datasets import cli as dataset_cli
    dataset_cli.add_parsers(sub)
    from phanet.learning import cli as learning_cli
    learning_cli.add_parsers(sub)
    from phanet.downstream import native_cli
    native_cli.add_parsers(sub)
    for name, description in (("discover-modules", "Predict modules, match references and report cross-genome annotation gaps"),
                              ("predict-modules", "Predict overlapping native module member sets"),
                              ("predict-go", "Predict GO terms using independent BP/CC/MF models"),
                              ("predict-vf", "Predict VF annotation labels with the fixed binary context model")):
        downstream = sub.add_parser(name, help=description)
        inputs = downstream.add_mutually_exclusive_group(required=True)
        inputs.add_argument("--input", type=Path, help="Nucleotide FASTA; run traditional preparation first")
        inputs.add_argument("--dataset", type=Path, help="Existing prepared/dataset directory with all native CDS")
        downstream.add_argument("--config", type=Path, help="Traditional annotation configuration for --input")
        downstream.add_argument("--model", type=Path, required=True, help="Downstream bundle; GO root contains BP/CC/MF")
        representations = downstream.add_mutually_exclusive_group()
        representations.add_argument("--embeddings", type=Path, help="Reuse verified full-length ESM2 store")
        representations.add_argument("--esm-model", type=Path, help="Local full-length ESM2 checkpoint")
        downstream.add_argument("--out", type=Path, required=True)
        downstream.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        downstream.add_argument("--batch-size", type=int, default=8)
        if name == "discover-modules":
            downstream.add_argument("--analysis-config", type=Path, required=True)
        if name == "predict-go":
            downstream.add_argument("--namespace", nargs="+", choices=["BP", "CC", "MF"], default=["BP", "CC", "MF"])
    analysis = sub.add_parser("analyze-modules", help="Analyze completed module predictions without reloading the model")
    analysis.add_argument("--dataset", type=Path, required=True)
    analysis.add_argument("--predictions", type=Path, required=True, help="Completed predict-modules output directory")
    analysis.add_argument("--config", type=Path, required=True, help="Bound module analysis configuration")
    analysis.add_argument("--out", type=Path, required=True)
    analysis_init = sub.add_parser("init-module-analysis", help="Bind local reference and annotation resources with SHA256")
    for field in ("references", "out"):
        analysis_init.add_argument("--" + field, type=Path, required=True)
    analysis_init.add_argument("--baseline-policy", type=Path, help="Default: bundled product-text-v3 policy (missing markers unresolved)")
    for field in ("reference-scope", "cluster-scope"):
        analysis_init.add_argument("--" + field, required=True)
    analysis_init.add_argument("--reference-format", choices=["native_members", "literature_targets"], default="native_members")
    for field in ("clusters", "profiles", "qualified-profiles", "sequence-alignments", "sequence-proteins", "baseline-decisions"):
        analysis_init.add_argument("--" + field, type=Path)
    for field in ("profile-scope", "sequence-scope"):
        analysis_init.add_argument("--" + field)
    for name, help_text in (("prepare", "Run Prodigal and eggNOG; export context and unresolved targets"),
                            ("doctor", "Check annotation tools and the installed eggNOG database")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", type=Path, help="JSON configuration (or set PHANET_CONFIG)")
        if name == "doctor":
            p.add_argument("--network-config", type=Path, default=os.environ.get("PHANET_NETWORK_CONFIG"), help="Also verify network tools and resources")
        if name == "prepare":
            p.add_argument("--input", required=True, type=Path, help="Nucleotide FASTA or FASTA.gz")
            p.add_argument("--out", required=True, type=Path)
            p.add_argument("--threads", type=int)
            p.add_argument("--resume", action="store_true", help="Reuse completed stages after identity and integrity checks")
    init = sub.add_parser("init-config", help="Write configuration for an existing annotation database")
    init.add_argument("--eggnog-data", required=True, type=Path)
    init.add_argument("--out", required=True, type=Path)
    init.add_argument("--prodigal", default="prodigal")
    init.add_argument("--emapper", nargs="+", default=["emapper.py"])
    network_init = sub.add_parser("init-network", help="Write a network resource configuration using local files")
    network_init.add_argument("--reference", type=Path, required=True)
    network_init.add_argument("--profiles", type=Path, required=True)
    network_init.add_argument("--tmvec-model", type=Path, required=True)
    network_init.add_argument("--prott5", type=Path, required=True)
    network_init.add_argument("--mmseqs", default="mmseqs")
    network_init.add_argument("--out", type=Path, required=True)
    setup = sub.add_parser("setup", help="Download, verify and unpack eggNOG 5.0.2; write configuration")
    setup.add_argument("--eggnog-data", type=Path, help="Database directory (default: ~/.local/share/phanet/eggnog/5.0.2)")
    setup.add_argument("--config", type=Path, default=Path("phanet.json"))
    setup.add_argument("--install-tools", action="store_true", help="Install annotation tools in an isolated Conda environment")
    setup.add_argument("--tools-prefix", type=Path, default=Path("~/.local/share/phanet/tools"))
    setup.add_argument("--manager", default="conda", help="Conda or Mamba executable")
    resources = sub.add_parser("setup-resources", help="Verify a manually downloaded PhaNet bundle and download official models")
    resources.add_argument("--bundle", type=Path, required=True)
    resources.add_argument("--external", type=Path, required=True)
    resources.add_argument("--out", type=Path, required=True, help="New configuration directory")
    resources.add_argument("--config", type=Path, required=True, help="Traditional configuration from phanet setup")
    resources.add_argument("--method", choices=["combined", "glm", "network"], default="combined")
    resources.add_argument("--mmseqs", default="mmseqs")
    resources.add_argument("--build-mmseqs", action="store_true", help="Download and build the exact compatible MMseqs2 source")
    resources.add_argument("--threads", type=int, default=4)
    resources.add_argument("--offline", action="store_true", help="Verify existing third-party files without downloading")
    infer = sub.add_parser("annotate", help="Traditional annotation followed by network and GLM supplementation")
    infer.add_argument("--method", choices=["combined", "glm", "network"], default="combined", help="Combined network and GLM (default), GLM only, or network only")
    infer.add_argument("--network-config", type=Path, default=os.environ.get("PHANET_NETWORK_CONFIG"), help="Network resource JSON (or PHANET_NETWORK_CONFIG)")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--out", type=Path, required=True)
    infer.add_argument("--config", type=Path)
    infer.add_argument("--model", type=Path, default=os.environ.get("PHANET_MODEL"), help="Inference bundle directory (or PHANET_MODEL)")
    infer.add_argument("--esm-model", type=Path, default=os.environ.get("PHANET_ESM_MODEL"), help="Local ESM2 checkpoint (or PHANET_ESM_MODEL)")
    infer.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    infer.add_argument("--batch-size", type=int, default=8)
    infer.add_argument("--threads", type=int)
    infer.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command in native_cli.COMMANDS:
            result = native_cli.run(args, progress=lambda msg: print(msg, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
            return 0
        if args.command in learning_cli.COMMANDS:
            result = learning_cli.run(args, progress=lambda msg: print(msg, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
            return 0
        if args.command in dataset_cli.COMMANDS:
            result = dataset_cli.run(args, progress=lambda msg: print(msg, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
            return 0
        if args.command in ("taxonomy", "rbp"):
            from phanet.downstream.evidence_cli import run
            print(json.dumps(run(args), indent=2))
            return 0
        if args.command == "init-module-analysis":
            from phanet.analysis.public import init_config
            values = vars(args).copy()
            values.pop("command")
            result = init_config(values.pop("out"), **values)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "analyze-modules":
            from phanet.analysis.public import analyze
            result = analyze(args.dataset, args.predictions, args.config, args.out,
                             progress=lambda msg: print(msg, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "discover-modules":
            from phanet.analysis.public import discover
            config_path = args.config or os.environ.get("PHANET_CONFIG")
            if args.input and not config_path:
                raise ValueError("Nucleotide input requires --config or PHANET_CONFIG")
            result = discover(args.out, args.model, args.analysis_config,
                dataset=args.dataset, input_fasta=args.input,
                config=PrepareConfig.load(config_path) if args.input else None,
                embeddings=args.embeddings, esm_checkpoint=args.esm_model,
                batch_size=args.batch_size, device=args.device,
                progress=lambda msg: print(msg, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
            return 0
        if args.command in ("predict-modules", "predict-go", "predict-vf"):
            from phanet.downstream.predict import predict
            config_path = args.config or os.environ.get("PHANET_CONFIG")
            if args.input and not config_path:
                raise ValueError("Nucleotide input requires --config or PHANET_CONFIG")
            task = {"predict-modules": "module", "predict-go": "go", "predict-vf": "vf"}[args.command]
            result = predict(task, args.out, args.model,
                             dataset=args.dataset, input_fasta=args.input,
                             config=PrepareConfig.load(config_path) if args.input else None,
                             embeddings=args.embeddings, esm_checkpoint=args.esm_model,
                             batch_size=args.batch_size, device=args.device,
                             namespaces=getattr(args, "namespace", ("BP", "CC", "MF")),
                             progress=lambda msg: print(msg, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "setup-resources":
            from phanet.resource_setup import setup_resources
            setup_resources(args.bundle, args.external, args.out, method=args.method,
                            traditional_config=args.config, mmseqs=args.mmseqs,
                            build=args.build_mmseqs, threads=args.threads, offline=args.offline,
                            progress=lambda msg: print(msg, file=sys.stderr, flush=True))
            return 0
        if args.command == "init-network":
            from phanet.network.inference import NetworkConfig
            values = {k: str(getattr(args, k).expanduser().resolve()) for k in ("reference", "profiles", "tmvec_model", "prott5")}
            values["mmseqs"] = str(Path(args.mmseqs).expanduser().resolve()) if "/" in args.mmseqs else args.mmseqs
            config = NetworkConfig(**values)
            with args.out.open("x") as stream:
                json.dump({"schema_version": 1, **asdict(config)}, stream, indent=2)
                stream.write("\n")
            print(f"Network configuration written to {args.out}; use doctor --network-config to verify resources")
            return 0
        if args.command == "setup":
            from phanet.resources import install_eggnog
            config_path = args.config.expanduser().absolute()
            existing = config_path.exists()
            if config_path.is_symlink():
                raise ValueError("Setup configuration must not be a symlink")
            if existing:
                config = PrepareConfig.load(config_path)
                if args.install_tools:
                    from phanet.tool_setup import tool_commands
                    expected = tool_commands(args.tools_prefix)
                    if config.prodigal != expected["prodigal"] or list(config.emapper) != expected["emapper"]:
                        raise ValueError("Existing configuration uses different tools; choose a new --config path")
                if args.eggnog_data and args.eggnog_data.expanduser().resolve() != Path(config.eggnog_data).resolve():
                    raise ValueError("Existing configuration uses a different database directory; choose a new --config path")
            else:
                if not config_path.parent.is_dir():
                    raise FileNotFoundError(f"Configuration parent directory does not exist: {config_path.parent}")
                root = args.eggnog_data or Path("~/.local/share/phanet/eggnog/5.0.2")
                config = PrepareConfig(str(root.expanduser().resolve()))
            if args.install_tools:
                from phanet.tool_setup import install_tools
                commands = install_tools(args.tools_prefix, manager=args.manager,
                                         progress=lambda msg: print(msg, file=sys.stderr, flush=True))
                config = replace(config, prodigal=commands["prodigal"], emapper=tuple(commands["emapper"]))
            # Validate configuration before starting a potentially large download.
            install_eggnog(config.eggnog_data, progress=lambda msg: print(msg, file=sys.stderr, flush=True))
            if not existing:
                with config_path.open("x") as stream:
                    json.dump({"schema_version": 1, **asdict(config)}, stream, indent=2)
                    stream.write("\n")
            print(f"Database ready; configuration: {config_path}. Run phanet doctor --config {config_path}")
            return 0
        if args.command == "init-config":
            def absolute_command(value):
                return str(Path(value).expanduser().resolve()) if "/" in value else value
            config = PrepareConfig(str(args.eggnog_data.expanduser().resolve()),
                                   prodigal=absolute_command(args.prodigal),
                                   emapper=tuple(absolute_command(v) for v in args.emapper))
            with args.out.open("x") as stream:
                json.dump({"schema_version": 1, **asdict(config)}, stream, indent=2)
                stream.write("\n")
            print(f"Configuration written to {args.out}; run phanet doctor to check resources")
            return 0
        path = args.config or os.environ.get("PHANET_CONFIG")
        if not path:
            parser.error("Provide --config or set PHANET_CONFIG to your resource configuration")
        config = PrepareConfig.load(path)
        if args.command == "annotate":
            from phanet.inference import annotate_glm
            if args.method != "network" and (args.model is None or args.esm_model is None):
                raise ValueError("Provide --model and --esm-model, or set PHANET_MODEL and PHANET_ESM_MODEL")
            if args.threads is not None:
                config = replace(config, threads=args.threads)
            import torch
            torch.set_num_threads(config.threads)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            network_config = None
            if args.method in ("combined", "network"):
                if args.network_config is None:
                    raise ValueError("Network inference requires --network-config")
                from phanet.network.inference import NetworkConfig
                network_config = NetworkConfig.load(args.network_config)
                print("Checking traditional tools before loading network resources", file=sys.stderr, flush=True)
                preflight(config)
            result = annotate_glm(args.input, args.out, config, args.model, args.esm_model,
                                  device=args.device, batch_size=args.batch_size, resume=args.resume, network_config=network_config,
                                  method=args.method,
                                  progress=lambda msg: print(msg, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
        elif args.command == "doctor":
            result = preflight(config)
            if args.network_config is not None:
                from phanet.network.inference import NetworkConfig, preflight as network_preflight
                _, result["network"] = network_preflight(NetworkConfig.load(args.network_config), progress=lambda msg: print(msg, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
        else:
            if args.threads is not None:
                config = replace(config, threads=args.threads)
            result = prepare(args.input, args.out, config, resume=args.resume,
                             progress=lambda msg: print(msg, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"phanet: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("phanet: interrupted; partial outputs retained", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
