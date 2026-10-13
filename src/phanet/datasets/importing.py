"""Import genomic records from annotated files or nucleotide FASTA."""
from pathlib import Path

from phanet.common.io import digest_file, write_json
from phanet.datasets.data import import_genbank, import_proteins
from phanet.datasets.schema import Dataset


def load_input(*, dataset=None, genbank=None, proteins=None, genes=None,
               genomes=None, input_fasta=None, config=None, work=None, progress=print, allow_empty=False):
    if sum(bool(x) for x in (dataset, genbank, proteins, input_fasta)) != 1:
        raise ValueError("Choose one input: dataset, GenBank, proteins or nucleotide FASTA")
    if proteins is not None and (genes is None or genomes is None):
        raise ValueError("Protein FASTA requires both genes and genomes tables")
    if proteins is None and (genes is not None or genomes is not None):
        raise ValueError("genes and genomes tables require protein FASTA")
    if dataset is not None:
        data = Dataset.load(dataset)
    elif genbank is not None:
        data = import_genbank(genbank)
    elif proteins is not None:
        data = import_proteins(proteins, genes, genomes)
    else:
        from phanet.annotation import PrepareConfig, prepare
        if config is None:
            raise ValueError("Nucleotide FASTA requires a traditional annotation configuration")
        prepare(input_fasta, Path(work) / "traditional", PrepareConfig.load(config), progress=progress)
        data = Dataset.load(Path(work) / "traditional/prepared/dataset")
    data.validate()
    if not data.genomes or (not data.proteins and not allow_empty):
        raise ValueError("Input must contain genomes and protein records")
    return data


def import_data(output, **inputs):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    data = load_input(work=out, **inputs)
    data.save(out / "dataset")
    result = {"genomes": len(data.genomes), "proteins": len(data.proteins),
              "files": {p.name: digest_file(p) for p in sorted((out / "dataset").iterdir())}}
    write_json(out / "summary.json", result)
    return result
