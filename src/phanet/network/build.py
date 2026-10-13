"""Build a three-channel reference network from a genomic dataset."""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import heapq
import importlib.metadata
import subprocess

import numpy as np
from scipy import sparse

from phanet.common.constants import CATEGORIES, CATEGORY_INDEX
from phanet.common.io import digest_file, write_json
from phanet.datasets.schema import Dataset
from phanet.network.resources import ARRAYS, NetworkBundle
from phanet.network.search import command, database_keys, write_fasta, profile_search, structure_edges
from phanet.representations.tmvec import full_length_vectors, model_identity


def reference_records(data):
    data.validate()
    sequences, labels = {}, defaultdict(set)
    for protein in data.proteins:
        sequences[protein.sha256] = protein.sequence
        if protein.labels:
            labels[protein.sha256].add(tuple(sorted(protein.labels)))
    if not sequences:
        raise ValueError("Network construction requires proteins")
    if any(len(values) > 1 for values in labels.values()):
        raise ValueError("Identical reference sequences have conflicting labels; resolve annotations before building")
    ids = sorted(sequences)
    bits = np.array([sum(1 << CATEGORY_INDEX[c] for c in next(iter(labels[h]), ())) for h in ids], np.uint16)
    if not bits.any():
        raise ValueError("Reference network requires annotated proteins")
    return dict((h, sequences[h]) for h in ids), bits


def sequence_layer(sequences, output, *, mmseqs, threads):
    out = Path(output); out.mkdir()
    ids = list(sequences); positions = {h: i for i, h in enumerate(ids)}
    write_fasta(out / "proteins.faa", sequences)
    cmd = lambda name, *args: command([mmseqs, *args], out, name)
    db = out / "proteins"
    cmd("createdb", "createdb", out / "proteins.faa", db, "--dbtype", 1, "--shuffle", 0, "--threads", threads)
    keys = database_keys(db)
    if set(keys) != set(ids):
        raise ValueError("Reference MMseqs identities differ from input proteins")
    (out / "diagonal.tsv").write_text("".join(f"{k}\t{k}\t0\t0\n" for k in sorted(keys.values())))
    cmd("diagonal", "tsv2db", out / "diagonal.tsv", out / "diagonal", "--output-dbtype", 7)
    cmd("self-align", "align", db, db, out / "diagonal", out / "self", "-e", "1e30",
        "--alignment-mode", 3, "-a", 1, "--threads", threads)
    cmd("self-convert", "convertalis", db, db, out / "self", out / "self.tsv",
        "--format-output", "query,target,bits", "--threads", threads)
    scores = {}
    for line in (out / "self.tsv").read_text().splitlines():
        q, r, bits = line.split(); value = float(bits)
        if q != r or q not in positions or q in scores or not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid reference self-alignment")
        scores[q] = value
    if set(scores) != set(ids):
        raise ValueError("Missing reference self-alignment scores")
    cmd("prefilter", "prefilter", db, db, out / "prefilter", "-s", "7.5", "-k", 6,
        "--max-seqs", len(ids), "--threads", threads, "--split-memory-limit", "8G")
    cmd("align", "align", db, db, out / "prefilter", out / "align", "-e", "1e-5",
        "--alignment-mode", 3, "-a", 1, "--threads", threads)
    cmd("convert", "convertalis", db, db, out / "align", out / "hits.tsv",
        "--format-output", "query,target,evalue,bits", "--threads", threads)
    neighbors = defaultdict(list)
    with (out / "hits.tsv").open() as stream:
        for line in stream:
            q, r, ev, bits = line.split(); ev, bits = float(ev), float(bits)
            if q not in positions or r not in positions or not np.isfinite([ev, bits]).all() or min(ev, bits) < 0:
                raise ValueError("Invalid reference alignment")
            if q == r or ev > 1e-5 or bits < 50:
                continue
            score = bits / max(scores[q], scores[r])
            if score > 1+1e-6:
                raise ValueError("Reference alignment exceeds its self-score normalization")
            if score >= .3:
                row = neighbors[positions[q]]
                item = (min(1., score), -positions[r])
                if len(row) < 200:
                    heapq.heappush(row, item)
                else:
                    heapq.heappushpop(row, item)
    rows, cols, values = [], [], []
    for q, hits in neighbors.items():
        for score, negative_index in hits:
            rows.append(q); cols.append(-negative_index); values.append(score)
    graph = sparse.csr_matrix((values, (rows, cols)), shape=(len(ids), len(ids)))
    return graph.maximum(graph.T), np.array([scores[h] for h in ids], np.float64)


def profile_layer(ids, hits):
    positions = {h: i for i, h in enumerate(ids)}
    families = defaultdict(dict)
    for hit in hits:
        p = positions[hit["protein_id"]]; coverage = float(hit["coverage"])
        if not np.isfinite(coverage) or not 0 < coverage <= 1:
            raise ValueError("Invalid profile coverage")
        family = families[hit["family_id"]]
        family[p] = max(family.get(p, 0), coverage)
    names = sorted(families)
    indices, coverage, ptr = [], [], [0]
    edges = {}
    for name in names:
        members = sorted(families[name].items(), key=lambda x: (-x[1], x[0]))
        indices.extend(p for p, _ in members); coverage.extend(c for _, c in members); ptr.append(len(indices))
        for q, qc in members:
            selected = [(r, rc) for r, rc in members[:51] if r != q][:50]
            for r, rc in selected:
                key = (q, r); edges[key] = max(edges.get(key, 0), min(qc, rc))
    pairs = list(edges)
    graph = sparse.csr_matrix(([edges[x] for x in pairs],
                               ([x[0] for x in pairs], [x[1] for x in pairs])), shape=(len(ids), len(ids)))
    arrays = {"profile_indices": np.array(indices, np.int64), "profile_indptr": np.array(ptr, np.int64),
              "profile_coverage": np.array(coverage, np.float64)}
    return graph.maximum(graph.T), names, arrays


def structure_layer(ids, vectors):
    bundle = SimpleNamespace(ids=np.array(ids, dtype="S64"), vectors=vectors)
    allowed = np.ones(len(ids), bool)
    positions = {h: i for i, h in enumerate(ids)}
    rows, cols, values = [], [], []
    for i, h in enumerate(ids):
        allowed[i] = False
        neighbors = structure_edges(bundle, {h: vectors[i]}, {}, allowed)[h]
        allowed[i] = True
        for r, score in neighbors.items():
            rows.append(i); cols.append(positions[r]); values.append(score)
    graph = sparse.csr_matrix((values, (rows, cols)), shape=(len(ids), len(ids)))
    return graph.maximum(graph.T)


def indexed_structure_layer(ids, vectors, backend):
    from phanet.network.graph import nearest_neighbors
    rows, cols, values = [], [], []
    for q, neighbors, scores in nearest_neighbors(vectors, backend=backend):
        for r, score in zip(neighbors, scores):
            if score >= .5:
                rows.append(q); cols.append(int(r)); values.append(float(score))
    graph = sparse.csr_matrix((values, (rows, cols)), shape=(len(ids), len(ids)))
    return graph.maximum(graph.T)


def merge_layers(layers):
    total = sparse.csr_matrix(layers[0].shape, dtype=np.float64)
    counts = total.copy()
    for layer in layers:
        layer = layer.tocsr().astype(np.float64)
        layer.eliminate_zeros(); layer.sort_indices()
        if np.any(~np.isfinite(layer.data)) or np.any((layer.data <= 0) | (layer.data > 1)):
            raise ValueError("Invalid network edge weights")
        total = total + layer
        mask = layer.copy(); mask.data[:] = 1
        counts = counts + mask
    total.sort_indices(); counts.sort_indices()
    total.data /= counts.data
    total.setdiag(0); total.eliminate_zeros(); total.sort_indices()
    return total


def reference_support(graph, labels):
    seed = np.column_stack([((labels[:, None] >> np.arange(9)) & 1), labels != 0]).astype(np.float64)
    old = seed.copy(); degree = np.asarray(graph.sum(axis=1)).ravel()
    for _ in range(200):
        walk = graph @ old
        new = .15*seed + .85*np.divide(walk, degree[:, None], out=np.zeros_like(walk), where=degree[:, None] > 0)
        new[degree == 0] = seed[degree == 0]
        residual = np.max(np.abs(new-old)); old = new
        if residual < 1e-7:
            return new
    raise RuntimeError("Reference propagation failed to converge")


def build_network(dataset, output, *, profiles, tmvec_model, prott5, mmseqs="mmseqs",
                  threads=4, device="cpu", structure_backend="exact", progress=print):
    try:
        import pyhmmer
    except ImportError as exc:
        raise RuntimeError("Install network dependencies: pip install 'phanet-core[network]'") from exc
    if threads < 1:
        raise ValueError("threads must be positive")
    if structure_backend not in ("exact", "faiss", "faiss-hnsw"):
        raise ValueError("Unknown structure search backend")
    if structure_backend != "exact":
        try:
            import faiss
            faiss.omp_set_num_threads(threads)
        except ImportError as exc:
            raise RuntimeError("Install indexed search: pip install 'phanet-core[faiss]'") from exc
    if pyhmmer.__version__ != "0.11.0":
        raise ValueError("Network construction requires PyHMMER 0.11.0")
    progress("Checking network tools and model files")
    version = subprocess.run([str(mmseqs), "version"], capture_output=True, text=True, check=True).stdout.strip()
    models = model_identity(tmvec_model, prott5)
    with pyhmmer.plan7.HMMFile(str(profiles)) as handle:
        count = sum(1 for _ in handle)
    if not count:
        raise ValueError("Profile library is empty")
    data = Dataset.load(dataset)
    sequences, labels = reference_records(data)
    ids = list(sequences)
    out = Path(output).expanduser().resolve(); out.mkdir(parents=True, exist_ok=False)
    bundle = out / "reference"; bundle.mkdir()
    work = out / "build"; work.mkdir()
    progress(f"Building reference network: {len(ids)} unique proteins")
    progress("Computing sequence relationships")
    sequence, self_bits = sequence_layer(sequences, work / "sequence", mmseqs=mmseqs, threads=threads)
    progress("Computing profile relationships")
    hits = profile_search(sequences, profiles, work / "profile", threads=threads, progress=progress,
                          profile_count=count, profile_Z=532910)
    profile, families, arrays = profile_layer(ids, hits)
    progress("Computing full-length structure representations")
    vectors = full_length_vectors(list(sequences.values()), tmvec_model, prott5, device=device, progress=progress)
    structure = (structure_layer(ids, vectors) if structure_backend == "exact"
                 else indexed_structure_layer(ids, vectors, structure_backend))
    progress("Merging network layers and propagating reference labels")
    graph = merge_layers([sequence, profile, structure])
    support = reference_support(graph, labels)
    arrays.update(ids=np.array(ids, dtype="S64"), labels=labels, self_bits=self_bits, vectors=vectors,
                  support=support, graph_data=graph.data, graph_indices=graph.indices, graph_indptr=graph.indptr)
    for name in ARRAYS:
        np.save(bundle / (name + ".npy"), arrays[name], allow_pickle=False)
    write_fasta(bundle / "references.faa", sequences)
    write_json(bundle / "families.json", families)
    meta = {"schema_version": 2, "method": "three-channel-fixed-label-ppr-v1", "categories": list(CATEGORIES),
            "embedding": "prott5-tmvec:full_length:float16:l2", "profile_Z": 532910, "profile_count": count,
            "phrog_sha256": digest_file(profiles), "mmseqs_version": version, "models": models,
            "structure_backend": structure_backend,
            "files": {p.name: {"bytes": p.stat().st_size, "sha256": digest_file(p)} for p in sorted(bundle.iterdir())}}
    write_json(bundle / "network.json", meta)
    NetworkBundle(bundle, progress=progress)
    config = {"schema_version": 1, "reference": "reference", "profiles": str(Path(profiles).expanduser().resolve()),
              "tmvec_model": str(Path(tmvec_model).expanduser().resolve()), "prott5": str(Path(prott5).expanduser().resolve()),
              "mmseqs": str(Path(mmseqs).expanduser().resolve()) if "/" in str(mmseqs) else str(mmseqs)}
    write_json(out / "network-config.json", config)
    summary = {"proteins": len(data.proteins), "references": len(ids), "edges": graph.nnz,
               "input_files": {p.name: digest_file(p) for p in sorted(Path(dataset).iterdir()) if p.is_file()},
               "network_sha256": digest_file(bundle / "network.json"),
               "pyhmmer": pyhmmer.__version__, "transformers": importlib.metadata.version("transformers")}
    write_json(out / "summary.json", summary)
    return summary
