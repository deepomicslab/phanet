"""HMMER signed scores and probability-field validation."""
import pytest
from phanet.network.graph import parse_hmmer_domtbl


@pytest.fixture
def domain_row():
    return ['protein', '-', '300', 'family', '-', '100', '1e-30', '100', '0',
            '2', '2', '0.5', '100', '-0.1', '0', '1', '50', '201', '250',
            '198', '253', '0.7', 'weak domain']


@pytest.mark.parametrize('score_column', [7, 13])
def test_weak_signed_domain_is_filtered(tmp_path, domain_row, score_column):
    domain_row[score_column] = '-0.1'
    path = tmp_path/'domains.tsv'; path.write_text(' '.join(domain_row)+'\n')
    assert parse_hmmer_domtbl(path) == []


@pytest.mark.parametrize('column,value', [(6, '-1'), (11, '-1'), (12, '-1'),
                                         (8, '-1'), (14, '-1'), (21, '1.1'),
                                         (21, '-0.1'), (7, 'nan'), (13, 'inf')])
def test_invalid_domain_fields_fail_before_filtering(tmp_path, domain_row, column, value):
    domain_row[column] = value
    path = tmp_path/'domains.tsv'; path.write_text(' '.join(domain_row)+'\n')
    with pytest.raises(ValueError): parse_hmmer_domtbl(path)
