"""Traditional annotation input and output tests."""
import pytest

from phanet.annotation import PrepareConfig, informative_product, normalize_fasta


@pytest.mark.parametrize("text", ["", "-", "hypothetical protein", "Protein of unknown function (DUF1327)"])
def test_unknown_descriptions_remain_targets(text):
    assert not informative_product(text)


def test_specific_description_can_be_kept_without_nine_class_mapping():
    assert informative_product("ribonuclease HII")


def test_fasta_identity_and_sequence_validation(tmp_path):
    source = tmp_path / "input.fasta"
    source.write_text(">phage|one original description\nacgtN\n>phage;two\nATGC\n")
    normalized = tmp_path / "normalized.fna"
    records = normalize_fasta(source, normalized)
    assert [r["contig_id"] for r in records] == ["phage|one", "phage;two"]
    assert normalized.read_text().startswith(">contig000001\nACGTN\n")
    source.write_text(">same\nACGT\n>same\nACGT\n")
    with pytest.raises(ValueError, match="duplicate"):
        normalize_fasta(source, normalized)


@pytest.mark.parametrize("kwargs", [{"threads": 0}, {"threads": True}, {"max_evalue": float("nan")}, {"emapper": "emapper.py"}])
def test_invalid_preparation_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PrepareConfig("database", **kwargs)
