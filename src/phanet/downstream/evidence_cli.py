"""Portable file adapters for RBP evidence and genome taxonomy analysis."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from phanet import __version__
from phanet.common.io import digest_file, read_table, write_json, write_table
from .evidence import feature_matrix, identifiers


def add_parsers(sub):
    taxonomy = sub.add_parser('taxonomy', help='Classify genome taxonomy from versioned hits and optional contextual features')
    taxonomy.add_argument('--queries', type=Path, required=True, help='One query genome ID per line, including no-hit genomes')
    taxonomy.add_argument('--hits', type=Path, help='TSV with genome_id, query_protein_id, weight and rank columns')
    taxonomy.add_argument('--reference-version', required=True, help='Taxonomy scheme and reference label/database version')
    for field in ('features', 'train-features', 'labels', 'config'):
        taxonomy.add_argument('--' + field, type=Path)
    taxonomy.add_argument('--feature-version', help='Shared encoder/pooling version of training and query NPZ features')
    taxonomy.add_argument('--out', type=Path, required=True)
    rbp = sub.add_parser('rbp', help='Summarize RBP computational evidence and domain annotations')
    actions = rbp.add_subparsers(dest='action', required=True)
    screen = actions.add_parser('screen', help='Screen supplied scores, retaining missing evidence states')
    screen.add_argument('--scores', type=Path, required=True, help='TSV keyed by protein_id; optional infection_score/classifier_score/tmvec_score/GO columns')
    for field in ('config', 'monomer', 'complex', 'literature', 'tmvec-query-vectors', 'tmvec-reference-vectors'):
        screen.add_argument('--' + field, type=Path)
    screen.add_argument('--tmvec-reference-version')
    screen.add_argument('--tmvec-top-k', type=int, default=50)
    domains = actions.add_parser('domains', help='Summarize supplied 0-based half-open domain intervals')
    domains.add_argument('--domains', type=Path, required=True)
    domains.add_argument('--protein-lengths', type=Path)
    for parser in (screen, domains):
        parser.add_argument('--evidence-version', required=True, help='Version/source of supplied scores or domain annotations')
        parser.add_argument('--out', type=Path, required=True)


def _lines(path):
    # Retain line positions so IDs align with array rows.
    return identifiers(Path(path).read_text(encoding='utf-8').splitlines())


def _features(path):
    data = np.load(path, allow_pickle=False)
    if not isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError('Features require an NPZ archive with ids and features arrays')
    with data:
        if not {'ids', 'features'} <= set(data.files):
            raise ValueError('Feature NPZ requires ids and features arrays')
        raw_ids = data['ids']
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in ('U', 'S'):
            raise ValueError('NPZ ids must be a one-dimensional string array')
        ids = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in raw_ids]
        return feature_matrix(data['features'], ids)


def _table(path, required):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        header = next(csv.reader(stream, delimiter='\t'), [])
    if len(header) != len(set(header)) or any(not key or key != key.strip() for key in header):
        raise ValueError('TSV columns must be nonempty, unique and without surrounding whitespace')
    return list(read_table(path, required))


def _config(path, constructor):
    if path is None:
        return constructor()
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Threshold configuration must be a JSON object')
    return constructor(**data)


def _version(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be explicitly identified')
    return value.strip()


def _taxonomy(args, out):
    from .taxonomy import TaxonomyThresholds, classify_taxonomy, fit_taxonomy_classifiers, predict_taxonomy_classifiers
    queries = _lines(args.queries)
    hits = _table(args.hits, ('genome_id', 'query_protein_id', 'weight')) if args.hits else []
    config = _config(args.config, TaxonomyThresholds)
    provided = [args.features is not None, args.train_features is not None, args.labels is not None]
    if any(provided) and not all(provided):
        raise ValueError('Contextual taxonomy requires --features, --train-features and --labels together')
    if not any(provided) and args.feature_version:
        raise ValueError('--feature-version requires contextual features')
    if not args.hits and not all(provided):
        raise ValueError('Supply taxonomy hits and/or complete contextual inputs')
    kwargs = dict(ranks=('order', 'family', 'subfamily', 'genus'))
    if all(provided):
        _version(args.feature_version, 'Feature encoder/pooling version')
        q, qids = _features(args.features)
        t, tids = _features(args.train_features)
        if set(qids) != set(queries) or set(qids) & set(tids) or q.shape[1] != t.shape[1]:
            raise ValueError('Taxonomy query/train features must be aligned, disjoint and dimension-compatible')
        labels = _table(args.labels, ('genome_id', 'partition'))
        train_ids = {row['genome_id'].strip() for row in labels if row['partition'] == 'train'}
        if not train_ids or train_ids != set(tids):
            raise ValueError('Training features must exactly cover the explicitly train-labeled genomes')
        models = fit_taxonomy_classifiers(t, tids, labels, train_ids)
        predictions = predict_taxonomy_classifiers(models, q, qids)
        kwargs = dict(labels=labels, features=np.vstack([t, q]), feature_ids=tids + qids,
                      train_ids=train_ids, classifier_predictions=predictions)
        serial = {}
        for rank, model in models.items():
            estimator = model['estimator']
            serial[rank] = {k: v for k, v in model.items() if k != 'estimator'}
            if estimator is not None:
                serial[rank].update(coefficients=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist(),
                                    iterations=estimator.n_iter_.tolist())
        write_json(out / 'fitted-heads.json', serial)
    result = classify_taxonomy(queries, hits, thresholds=config, **kwargs)
    write_json(out / 'predictions.json', result)
    rows = [dict(genome_id=row['genome_id'], rank=rank, label=value['label'],
                 confidence=value['confidence'], method=value['method'])
            for row in result for rank, value in row['ranks'].items()]
    write_table(out / 'predictions.tsv', rows, ['genome_id', 'rank', 'label', 'confidence', 'method'])
    return dict(task='taxonomy', queries=len(queries), thresholds=asdict(config),
                reference_version=args.reference_version, feature_version=args.feature_version,
                interpretation='Similarity confidence is summed supporting weight divided by informative query-protein count; centroid confidence uses relative squared distance.')


def _rbp(args, out):
    from .rbp import RBPScreenConfig, domain_architecture, screen_rbp, tmvec_retrieve
    if args.action == 'domains':
        lengths = None
        if args.protein_lengths:
            table = _table(args.protein_lengths, ('protein_id', 'length'))
            ids = identifiers([r['protein_id'] for r in table])
            lengths = {pid: row['length'] for pid, row in zip(ids, table)}
        rows = _table(args.domains, ('protein_id', 'start', 'end'))
        result = domain_architecture(rows, protein_lengths=lengths)
        write_json(out / 'report.json', result)
        write_table(out / 'domains.tsv', result['observations'], ['protein_id', 'domain', 'start', 'end', 'normalized_center'])
        return dict(task='rbp-domains', domains=result['domains'], instances=result['instances'])
    rows = _table(args.scores, ('protein_id',))
    if not rows:
        raise ValueError('RBP score table must contain at least one protein')
    ids = identifiers([row['protein_id'] for row in rows])
    config = _config(args.config, RBPScreenConfig)
    terms = [key for key in rows[0] if key.startswith('GO:')]

    def column(key):
        return np.asarray([np.nan if row[key].strip() == '' else float(row[key]) for row in rows]) if key in rows[0] else None

    tm = column('tmvec_score')
    retrieval_args = [args.tmvec_query_vectors, args.tmvec_reference_vectors]
    if any(v is not None for v in retrieval_args):
        if not all(v is not None for v in retrieval_args) or tm is not None:
            raise ValueError('Supply both TM-Vec vector files, without a tmvec_score column')
        _version(args.tmvec_reference_version, 'TM-Vec reference/checkpoint version')
        q, qids = _features(args.tmvec_query_vectors)
        r, rids = _features(args.tmvec_reference_vectors)
        if set(qids) != set(ids):
            raise ValueError('TM-Vec query IDs must exactly match the RBP score table')
        index = {pid: i for i, pid in enumerate(qids)}
        retrieval = tmvec_retrieve(ids, q[[index[pid] for pid in ids]], rids, r, top_k=args.tmvec_top_k)
        tm = retrieval['best_scores']
        write_table(out / 'tmvec_retrieval.tsv', retrieval['pairs'], ['protein_id', 'reference_id', 'cosine_score', 'rank'])
    elif args.tmvec_reference_version:
        raise ValueError('--tmvec-reference-version requires TM-Vec vector inputs')
    if not terms and tm is None and 'infection_score' not in rows[0]:
        raise ValueError('RBP screening needs preliminary GO, TM-Vec or Infection evidence')
    result = screen_rbp(ids, go_scores=np.column_stack([column(term) for term in terms]) if terms else None,
        go_terms=terms, tmvec_scores=tm, classifier_scores=column('classifier_score'),
        infection_scores=column('infection_score'), config=config,
        monomer_rows=_table(args.monomer, ('protein_id', 'mean_plddt')) if args.monomer else None,
        complex_rows=_table(args.complex, ('protein_id', 'iptm', 'buried_surface_area', 'tm_score')) if args.complex else None,
        literature_ids=_lines(args.literature) if args.literature else None)
    write_json(out / 'report.json', result)
    write_table(out / 'candidates.tsv', result['candidates'], list(result['candidates'][0]))
    return dict(task='rbp-screen', library=result['library_size'], stages=result['stages'],
                thresholds=asdict(config), preliminary_rule=result['preliminary_rule'])


def run(args):
    """Run file-based analysis and write results, provenance and execution status."""
    version_name = 'reference_version' if args.command == 'taxonomy' else 'evidence_version'
    _version(getattr(args, version_name), version_name)
    out = args.out.expanduser().absolute()
    if out.is_symlink() or out.exists():
        raise FileExistsError('Choose a new output directory; previous attempts are retained')
    inputs = {}
    settings = {}
    for key, value in vars(args).items():
        if key == 'out':
            continue
        if isinstance(value, Path):
            path = value.expanduser().resolve(strict=True)
            setattr(args, key, path)
            inputs[key] = dict(path=str(path), sha256=digest_file(path))
        else:
            settings[key] = value
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'status.json', dict(state='running', task=args.command))
    try:
        import scipy
        import sklearn
        import sys
        provenance = dict(schema_version=1, package_version=__version__,
            created_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs, settings=settings,
            environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))
        source = out / 'source'
        source.mkdir()
        paths = [Path(__file__).with_name(name) for name in ('rbp.py', 'taxonomy.py', 'evidence.py', 'evidence_cli.py')]
        paths += [Path(__file__).parents[1] / 'cli.py', Path(__file__).parents[1] / 'common' / 'io.py']
        provenance['source'] = {}
        for path in paths:
            shutil.copyfile(path, source / path.name)
            provenance['source'][path.name] = digest_file(path)
        write_json(out / 'provenance.json', provenance)
        summary = _taxonomy(args, out) if args.command == 'taxonomy' else _rbp(args, out)
        for item in inputs.values():
            if digest_file(item['path']) != item['sha256']:
                raise ValueError('Input changed during analysis; outputs are not accepted')
        write_json(out / 'summary.json', summary)
        outputs = {str(p.relative_to(out)): digest_file(p) for p in sorted(out.rglob('*'))
                   if p.is_file() and p.name != 'status.json'}
        write_json(out / 'manifest.json', dict(schema_version=1, task=args.command, inputs=inputs, outputs=outputs, status='complete'))
        write_json(out / 'status.json', dict(state='complete', task=args.command))
        return summary
    except BaseException as exc:
        write_json(out / 'status.json', dict(state='failed', task=args.command, error=f'{type(exc).__name__}: {exc}'))
        raise
