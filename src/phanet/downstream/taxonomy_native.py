"""Hierarchical taxonomy transfer, parent-masked heads and genome pooling."""
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from phanet.downstream.taxonomy import TaxonomyThresholds
from phanet.downstream.evidence import identifier, number

RANKS = ('family','subfamily','genus')


def best_hits(hits, *, ranks=RANKS):
    fields = tuple(dict.fromkeys(('order', *ranks)))
    best, owners = {}, {}
    for raw in hits:
        row = dict(raw)
        gid, pid = identifier(row.get('genome_id')), identifier(row.get('query_protein_id'))
        if pid in owners and owners[pid] != gid:
            raise ValueError('Query protein belongs to multiple genomes')
        owners[pid] = gid
        row['weight'] = number(row.get('weight'), 'taxonomy hit weight', high=1.)
        for rank in fields:
            value = row.get(rank, '')
            if value is None: value = ''
            if not isinstance(value, str): raise ValueError('Taxonomy labels must be strings')
            row[rank] = value.strip()
        if not row['weight'] or not any(row[r] for r in ranks):
            continue
        key = (-row['weight'], tuple(row[r] for r in fields), str(row.get('reference_id','')))
        if (gid,pid) not in best or key < best[gid,pid][0]:
            best[gid,pid] = (key,row)
    grouped = defaultdict(list)
    for (gid,pid), (_,row) in sorted(best.items()): grouped[gid].append(row)
    return grouped


def transfer(query_ids, hits, thresholds=None, *, ranks=('order', *RANKS)):
    from .taxonomy import _ranks
    ranks = _ranks(ranks)
    levels = (*RANKS, 'species')
    last = max((levels.index(r) for r in ranks if r != 'order'), default=0)
    levels = levels[:last+1]
    thresholds = thresholds or TaxonomyThresholds()
    query_ids = list(query_ids)
    if not query_ids or len(set(query_ids)) != len(query_ids): raise ValueError('Unique query genomes are required')
    for q in query_ids: identifier(q)
    grouped = best_hits(hits, ranks=(*RANKS, 'species') if 'species' in levels else RANKS)
    if set(grouped)-set(query_ids): raise ValueError('Taxonomy hits contain unknown query genomes')
    output = []
    for gid in query_ids:
        rows = grouped[gid]
        result = {r:dict(label='',confidence=None,method='unresolved') for r in ('order',*levels)}
        accepted = {}
        for rank in levels:
            informative = [r for r in rows if r[rank]]
            if rank == 'family':
                informative += [r for r in rows if not r['family'] and any(r.get(k) for k in levels[1:])]
            if rank == 'subfamily':
                if not any(r['subfamily'] for r in rows if r['family'] == accepted.get('family')):
                    continue
                informative += [r for r in rows if not r['subfamily'] and r.get('genus')]
            candidates = defaultdict(list)
            for row in informative:
                if all(row[p] == label for p,label in accepted.items()):
                    candidates[row[rank]].append(row)
            if not candidates: break
            h = len(informative)
            ordered = sorted(candidates, key=lambda x:(-sum(r['weight'] for r in candidates[x])/h,x))
            label = ordered[0]; selected = candidates[label]
            confidence = sum(r['weight'] for r in selected)/h
            runner = sum(r['weight'] for r in candidates[ordered[1]])/h if len(ordered)>1 else 0.
            limits = thresholds.for_rank(rank)
            if (len(selected)<limits['min_support'] or len(selected)/h<limits['min_proportion']
                    or confidence<limits['min_confidence'] or confidence-runner<limits['min_margin']): break
            accepted[rank] = label
            if not label:
                continue
            result[rank] = dict(label=label,confidence=confidence,margin=confidence-runner,
                support_count=len(selected),support_proportion=len(selected)/h,query_count=h,method='similarity_aggregate')
            if label:
                orders = {r['order'] for r in selected if r['order']}
                if len(orders)>1: raise ValueError('Taxon has conflicting order ancestry')
                if orders: result['order'] = dict(label=next(iter(orders)),confidence=confidence,method='similarity_aggregate')
        selected = {rank:result[rank] for rank in ranks}
        output.append(dict(genome_id=gid,ranks=selected,
                           method='similarity_aggregate' if any(r['label'] for r in selected.values()) else 'ambiguous'))
    return output


def _label(row, rank):
    value = row.get(rank)
    if value is None: value = ''
    if not isinstance(value, str): raise ValueError('Taxonomy labels must be strings')
    return value.strip()


def _merge_ancestry(target, source):
    for rank, value in source.items():
        if value and target.get(rank) and target[rank] != value:
            raise ValueError('Conflicting taxonomy ancestry')
        if value: target[rank] = value


def taxonomy_tree(labels, minimum_parents=5):
    if minimum_parents < 1: raise ValueError('Minimum parent count must be positive')
    support, paths = defaultdict(set), {}
    seen = set()
    for row in labels:
        if row['partition'] != 'train': continue
        gid = identifier(row['genome_id'])
        if gid in seen: raise ValueError('Duplicate training genome label')
        seen.add(gid)
        parent = identifier(row.get('parent_id') or gid)
        for rank in RANKS:
            label = _label(row, rank)
            if not label: continue
            path = {r:_label(row,r) for r in ('order',*RANKS[:RANKS.index(rank)])}
            _merge_ancestry(paths.setdefault((rank,label), dict.fromkeys(path,'')), path)
            support[rank,label].add(parent)
    # Share only known training ancestry; empty cells provide no conflicting evidence.
    changed = True
    while changed:
        before = {key:dict(path) for key,path in paths.items()}
        for (rank,label),path in paths.items():
            for ancestor in RANKS[:RANKS.index(rank)]:
                upper = paths.get((ancestor,path[ancestor]))
                if upper is not None:
                    _merge_ancestry(upper, {r:path[r] for r in upper})
                    _merge_ancestry(path, upper)
        changed = paths != before
    tree = {r:{label:paths[r,label] for rr,label in sorted(paths)
               if rr == r and len(support[rr,label])>=minimum_parents} for r in RANKS}
    if not any(tree.values()): raise ValueError('No training-supported taxonomy classes')
    return tree


class TaxonomyModel(nn.Module):
    def __init__(self, encoder, tree):
        super().__init__()
        if set(tree)!=set(RANKS) or not any(tree.values()):
            raise ValueError('Invalid taxonomy tree ranks')
        for rank in RANKS:
            for name,path in tree[rank].items():
                if not isinstance(name,str) or not name.strip() or not isinstance(path,dict):
                    raise ValueError('Invalid taxonomy node')
                expected={'order',*RANKS[:RANKS.index(rank)]}
                if set(path)!=expected or any(not isinstance(v,str) for v in path.values()):
                    raise ValueError('Invalid taxonomy ancestry')
                for ancestor in RANKS[:RANKS.index(rank)]:
                    upper=tree[ancestor].get(path[ancestor],{})
                    if any(value and path[k] and path[k]!=value for k,value in upper.items()):
                        raise ValueError('Conflicting taxonomy ancestry')
        self.encoder, self.tree = encoder, tree
        self.classes = {r:sorted(tree[r]) for r in RANKS}
        # Extra logits route to descendants whose parent rank has no retained class.
        orphan = any(self.branch(path,'family') == '' for r in ('subfamily','genus') for path in tree[r].values())
        self.family_direct = len(self.classes['family']) if orphan else None
        subfamilies = {self.branch(path,'family') for path in tree['subfamily'].values()}
        direct = sorted({self.branch(path,'family') for path in tree['genus'].values()
                         if not self.branch(path,'subfamily')} & subfamilies)
        self.direct_branches = {family:len(self.classes['subfamily'])+i for i,family in enumerate(direct)}
        sizes = {r:len(self.classes[r]) for r in RANKS}
        sizes['family'] += int(orphan)
        sizes['subfamily'] += len(direct)
        self.heads = nn.ModuleDict({r:nn.Linear(encoder.config.hidden_dim,n) for r,n in sizes.items() if n})

    def branch(self, path, rank):
        value=path.get(rank,'')
        return value if value in self.tree[rank] else ''

    def loss(self, vector, label):
        losses = []
        known = {r:_label(label,r) for r in RANKS}
        descendant = next((self.tree[r][known[r]] for r in reversed(RANKS)
                           if known[r] in self.tree[r]), None)
        for rank in RANKS:
            name=known[rank]
            if name in self.classes[rank]:
                target=self.classes[rank].index(name)
                path=self.tree[rank][name]
            elif (rank=='family' and descendant is not None
                  and not self.branch(descendant,'family') and known['family'] not in self.tree['family']):
                target=self.family_direct; path=descendant
                if target is None: continue
            elif (rank=='subfamily' and known['genus'] in self.tree['genus']
                  and not self.branch(self.tree['genus'][known['genus']],'subfamily')
                  and self.branch(self.tree['genus'][known['genus']],'family') in self.direct_branches):
                path=self.tree['genus'][known['genus']]
                target=self.direct_branches[self.branch(path,'family')]
            else:
                continue
            parents={r:self.branch(path,r) for r in RANKS[:RANKS.index(rank)]}
            candidates=self.candidates(rank,parents)
            if target not in candidates: raise ValueError('Taxonomy target is excluded by its ancestry')
            logits=self.heads[rank](vector)[candidates]
            losses.append(nn.functional.cross_entropy(logits[None],torch.tensor([candidates.index(target)],device=vector.device)))
        return torch.stack(losses).mean() if losses else None

    def candidates(self, rank, parents):
        if rank not in self.heads: return []
        candidates = []
        for i,label in enumerate(self.classes[rank]):
            path = self.tree[rank][label]
            if any(self.branch(path,r)!=parents.get(r,'') for r in RANKS[:RANKS.index(rank)]): continue
            candidates.append(i)
        if rank=='family' and self.family_direct is not None: candidates.append(self.family_direct)
        if rank == 'subfamily' and parents.get('family','') in self.direct_branches:
            candidates.append(self.direct_branches[parents.get('family','')])
        return candidates

    def classify(self, vector, thresholds):
        assigned = {}; rows = {r:dict(label='',confidence=None,method='unresolved') for r in ('order',*RANKS)}
        for rank in RANKS:
            candidates = self.candidates(rank,assigned)
            if not candidates: continue
            probabilities = self.heads[rank](vector)[candidates].softmax(-1)
            position = int(probabilities.argmax()); confidence = float(probabilities[position])
            if confidence < thresholds[rank]: break
            if candidates[position] >= len(self.classes[rank]):
                assigned[rank] = ''
                rows[rank] = dict(label='', confidence=confidence, method='context_glm')
                continue
            name = self.classes[rank][candidates[position]]; assigned[rank] = name
            rows[rank] = dict(label=name,confidence=confidence,method='context_glm')
            if self.tree[rank][name].get('order') and not rows['order']['label']:
                rows['order'] = dict(label=self.tree[rank][name]['order'],confidence=confidence,method='context_glm')
        return rows
