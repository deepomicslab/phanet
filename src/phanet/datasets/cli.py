"""Commands for importing data, preparing training inputs and building networks."""
from pathlib import Path
import os


COMMANDS = {"import-data", "prepare-training", "build-network"}


def add_parsers(sub):
    for name, description in (("import-data", "Import genomic records"),
                              ("prepare-training", "Prepare GLM training data")):
        parser = sub.add_parser(name, help=description)
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--dataset", type=Path)
        source.add_argument("--genbank", type=Path, nargs="+")
        source.add_argument("--proteins", type=Path, help="Protein FASTA with genes and genomes tables")
        source.add_argument("--input", type=Path, help="Nucleotide FASTA")
        parser.add_argument("--genes", type=Path)
        parser.add_argument("--genomes", type=Path)
        parser.add_argument("--config", type=Path, help="Traditional annotation configuration for nucleotide FASTA")
        parser.add_argument("--out", type=Path, required=True)
        if name == "prepare-training":
            features = parser.add_mutually_exclusive_group(required=True)
            features.add_argument("--embeddings", type=Path, help="Existing ESM2 EmbeddingStore")
            features.add_argument("--esm-checkpoint", type=Path)
            parser.add_argument("--splits", type=Path, help="Existing annotation SplitManifest JSON")
            parser.add_argument("--ratios", type=float, nargs=3, default=(.8, .1, .1), metavar=("TRAIN", "VAL", "TEST"))
            parser.add_argument("--unit", choices=("genome", "cluster"), default="genome")
            parser.add_argument("--seed", type=int, default=7)
            parser.add_argument("--max-context", type=int, default=30)
            parser.add_argument("--device", default="cpu")
            parser.add_argument("--batch-size", type=int, default=1)
            family = parser.add_mutually_exclusive_group()
            family.add_argument("--family-hits", type=Path)
            family.add_argument("--profiles", type=Path)
            parser.add_argument("--threads", type=int, default=4)
    network = sub.add_parser("build-network", help="Build a custom reference network")
    network.add_argument("--dataset", type=Path, required=True)
    network.add_argument("--profiles", type=Path, required=True)
    network.add_argument("--tmvec-model", type=Path, required=True)
    network.add_argument("--prott5", type=Path, required=True)
    network.add_argument("--mmseqs", default="mmseqs")
    network.add_argument("--threads", type=int, default=4)
    network.add_argument("--device", default="cpu")
    network.add_argument("--structure-backend", choices=("exact", "faiss", "faiss-hnsw"), default="exact")
    network.add_argument("--out", type=Path, required=True)


def run(args, *, progress=print):
    if args.command == "build-network":
        from phanet.network.build import build_network
        return build_network(args.dataset, args.out, profiles=args.profiles, tmvec_model=args.tmvec_model,
                             prott5=args.prott5, mmseqs=args.mmseqs, threads=args.threads,
                             device=args.device, structure_backend=args.structure_backend, progress=progress)
    inputs = dict(dataset=args.dataset, genbank=args.genbank, proteins=args.proteins,
                  genes=args.genes, genomes=args.genomes, input_fasta=args.input,
                  config=args.config or os.environ.get("PHANET_CONFIG"), progress=progress)
    if args.command == "import-data":
        from phanet.datasets.importing import import_data
        return import_data(args.out, **inputs)
    from phanet.learning.data import prepare_training
    return prepare_training(args.out, embeddings=args.embeddings, esm_checkpoint=args.esm_checkpoint,
                            splits=args.splits, ratios=args.ratios, unit=args.unit, seed=args.seed,
                            max_context=args.max_context, device=args.device, batch_size=args.batch_size,
                            family_hits=args.family_hits, profiles=args.profiles, threads=args.threads, **inputs)
