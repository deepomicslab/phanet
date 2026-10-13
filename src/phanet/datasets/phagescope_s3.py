"""PhageScope Table S3 vocabulary and product-keyword matching."""
from functools import lru_cache
import re

from phanet.common.io import digest_object

VERSION = "phagescope-s3-keyword-v1"
SOURCE = "PhageScope gkad979 Supplementary Methods S1.3 and Table S3"
PDF_SHA256 = "ad7ec1a9332675ff028d189be68b5c0439fccd822e5be1c6a66baf7a150eb8c7"

KEYWORDS = {'Lysis': ('lysis',
           'holin',
           'lysozy',
           'lysin',
           'lys',
           'lysozyme',
           'lyzozyme',
           'kil',
           'rll',
           'rz',
           'rz1',
           'protease',
           'peptidase',
           'lytic',
           'lyase',
           'Panton-Valentine'),
 'Integration': ('integration',
                 'integrase',
                 'recombinase',
                 'excise',
                 'excisionase',
                 'recombination',
                 'beta',
                 'cre',
                 'transposase',
                 'lysogenic'),
 'Replication': ('replication',
                 'polymerase',
                 'helicase',
                 'ligase',
                 'primase',
                 'dna synthesis',
                 'clamp',
                 'uvs',
                 'binding',
                 'helix',
                 'topoisomerase',
                 'reductase',
                 'cell division'),
 'tRNA-related': ('trna',),
 'Regulation': ('regulator',
                'inhibition',
                'operator',
                'activator',
                'inhibitor',
                'repressor',
                'transcription',
                'modulator',
                'c1',
                'c2',
                'whib',
                'regulatory',
                'helix',
                'cro',
                'anti-adapter',
                'anti-termination',
                'antitermination',
                'methylase',
                'translation initiation',
                'translation regulation'),
 'Packaging': ('terminase',
               'package',
               'packaging',
               'endonuclease',
               'exonuclease',
               'hnh',
               'maturase',
               'nuclease'),
 'Assembly': ('head',
              'capsid',
              'coat',
              'gp',
              'collar',
              'scaffold',
              'core',
              'neck',
              'membrane',
              'tapemeasure',
              'tape measure',
              'cp12',
              'structural',
              'thioredoxi',
              'particle',
              'assembly',
              'cell envelope',
              'cell surface',
              'cell wall',
              'circularization'),
 'Infection': ('tail',
               'virion',
               'plate',
               'chaperone',
               'fiber',
               'cell wall hydrolase',
               'a2',
               'a1',
               'injection',
               'infection',
               'portal',
               'vertex',
               'attachment',
               'spike',
               'maturation',
               'anti-CRISPR',
               'anti-restriction',
               'antirestriction',
               'transporter',
               'host-specificity',
               'host specificity',
               'host attachement',
               'host range',
               'host interaction',
               'adhesion'),
 'Immune': ('transferase', 'immunity', 'immunodominant', 'immunoglobulin'),
 'Hypothetical': ('hypothetical',)}
CATEGORIES = tuple(c for c in KEYWORDS if c != "Hypothetical")
MATCHING = {
    "normalization": "casefold; non-ASCII-alphanumeric runs become one space; trim",
    "short_keywords": "normalized length <=3: complete alphanumeric token",
    "other_keywords": "literal normalized substring; no additional synonyms or stemming",
    "overlap": "retain every matching category; no category priority or enzyme exclusion",
    "hypothetical": "retain status and keyword evidence, exclude from functional vector",
    "unmatched": "empty functional set; not assigned Hypothetical",
}


def normalize(text):
    if not isinstance(text, str):
        raise TypeError("product description must be text")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def native_categories(value):
    """None means missing classification; () means an explicit unresolved state."""
    if value is None or (isinstance(value, str) and not value.strip(' ;\t\r\n')):
        return None
    if not isinstance(value, str):
        raise TypeError('native classification must be text or None')
    names = {c.casefold(): c for c in CATEGORIES}
    tokens = {t.strip().casefold().replace('_', '-') for t in value.split(';') if t.strip()}
    unknown = tokens - set(names) - {'hypothetical', 'unsorted'}
    if unknown:
        raise ValueError('unknown native classification: ' + ', '.join(sorted(unknown)))
    return tuple(c for c in CATEGORIES if c.casefold() in tokens)


class PhageScopeS3Mapper:
    categories = CATEGORIES
    version = VERSION

    def __init__(self):
        self._patterns = {
            category: tuple((term, re.compile(
                r"(?<![a-z0-9])" + re.escape(normalize(term)) + r"(?![a-z0-9])"
                if len(normalize(term)) <= 3 else re.escape(normalize(term))))
                for term in terms)
            for category, terms in KEYWORDS.items()
        }

    @property
    def fingerprint(self):
        return digest_object(dict(version=VERSION, categories=self.categories,
                                  keywords=KEYWORDS, matching=MATCHING, pdf_sha256=PDF_SHA256))

    @lru_cache(maxsize=65536)
    def matches(self, text):
        term = normalize(text)
        return tuple((category, tuple(word for word, pattern in patterns if pattern.search(term)))
                     for category, patterns in self._patterns.items()
                     if any(pattern.search(term) for _, pattern in patterns))

    def map(self, text):
        present = dict(self.matches(text))
        return tuple(c for c in self.categories if c in present)

    def explain(self, text):
        hits = dict(self.matches(text))
        functional = self.map(text)
        return dict(text=text, normalized=normalize(text), categories=functional,
                    matches=hits, hypothetical="Hypothetical" in hits,
                    state="functional" if functional else "hypothetical" if "Hypothetical" in hits else "unmapped")
