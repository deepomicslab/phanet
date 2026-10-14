# Resource sources and terms

The PhaNet source code uses the MIT license. Separate datasets and upstream models
retain their own terms; PhaNet's MIT license does not replace them. The manual resource
archive includes a `RESOURCE_NOTICES.md` explaining its contents. The code package does
not contain checkpoints or reference databases.

| Resource | Distribution and source |
|---|---|
| PhaNet GLM | Annotation bundle or [model installer](models.md); inference weights and model metadata. |
| PhaNet Module / GO / VF / RBP / taxonomy | [Model folder](https://drive.google.com/drive/folders/1RajAiCH51hHKPcjBYXlXDyw3FjhAHxqq); install with `phanet setup-models`. |
| PhaNet reference network | Manual bundle; fixed derived graph, functional labels, reference sequences and representations. Source sequence/data terms remain applicable. |
| Converted PHROG v4 profiles | Manual bundle; 38,880 official PHROG alignments converted with HMMER 3.4 `hmmbuild --amino`. The HMMER file is identified by SHA-256 in the network manifest. |
| ESM2 650M | Automatic download from [Meta ESM](https://github.com/facebookresearch/esm), including the contact-regression companion. |
| ProtT5-XL-UniRef50 | Automatic download from [Rostlab on Hugging Face](https://huggingface.co/Rostlab/prot_t5_xl_uniref50), revision `973be27c52ee6474de9c945952a8008aeb2a1a73`. |
| TM-Vec CATH-large | Automatic original checkpoint download from [TMvec/DeepBLAST models](https://doi.org/10.6084/m9.figshare.25810099), file `49181533`. The Figshare record specifies CC BY 4.0. |
| MMseqs2 | Automatic source build from [MMseqs2](https://github.com/soedinglab/MMseqs2) commit `fed69f84d98a2f419693c3b41e0bdb5ad60f7aaa`. Upstream GPL-3.0 terms apply. |
| eggNOG | Automatic database download from the [eggNOG-mapper](https://github.com/eggnogdb/eggnog-mapper) versioned database server, database 5.0.2. |
| Prodigal / eggNOG-mapper / DIAMOND | Optional isolated Conda installation from the upstream packages; package licenses remain applicable. |
| Example contigs | Public ENA accessions [J02459.1](https://www.ebi.ac.uk/ena/browser/view/J02459.1) and [LC889025.1](https://www.ebi.ac.uk/ena/browser/view/LC889025.1); coordinates and SHA-256 are included beside the FASTA files. |

PHROG source: [PHROG database](https://phrogs.lmge.uca.fr/), [Terzian et al.,
2021](https://doi.org/10.1093/nargab/lqab067). The PHROG HMMER conversion uses HMMER 3.4
and is identified by SHA-256 in the network manifest.

Reference data and class mapping build on
[PhageScope](https://doi.org/10.1093/nar/gkad979), [NCBI
RefSeq](https://www.ncbi.nlm.nih.gov/refseq/) and PHROG. ESM2: [Lin et al.,
2023](https://doi.org/10.1126/science.ade2574). ProtTrans: [Elnaggar et
al.](https://doi.org/10.1109/TPAMI.2021.3095381). TM-Vec implementation attribution and
its BSD-3-Clause text are included in
[THIRD_PARTY_NOTICES.txt](../THIRD_PARTY_NOTICES.txt).

All automatic model downloads are pinned by full SHA-256 and file size in the installer.
Use `phanet setup-resources` for external models and annotation configuration,
and `phanet setup-models` for PhaNet prediction bundles.
