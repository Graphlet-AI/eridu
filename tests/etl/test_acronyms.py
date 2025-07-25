import pytest

from eridu.etl.acronyms import (
    get_corporate_ending,
    get_multilingual_stop_words,
    process_single_name,
)


def test_get_multilingual_stop_words():
    """Test that stop words are retrieved correctly"""
    stop_words = get_multilingual_stop_words()
    assert isinstance(stop_words, set)
    assert len(stop_words) > 0
    # Check for common English stop words
    assert "the" in stop_words
    assert "and" in stop_words
    assert "of" in stop_words
    assert "es" in stop_words


def test_get_corporate_ending():
    """Test extraction of corporate endings"""
    # Test with various company names
    assert get_corporate_ending("Apple Inc.") == "Inc."
    assert get_corporate_ending("Microsoft Corporation") == "Corporation"
    assert get_corporate_ending("Google LLC") == "LLC"
    assert get_corporate_ending("IBM") == ""
    assert get_corporate_ending("") == ""
    assert get_corporate_ending(None) == ""


@pytest.mark.parametrize(
    "company_name,expected_abbreviations",
    [
        (
            "International Business Machines Corporation",
            [
                {"original": "International Business Machines Corporation", "abbreviated": "IBM"},
                {
                    "original": "International Business Machines Corporation",
                    "abbreviated": "IBM Corporation",
                },
                {
                    "original": "International Business Machines Corporation",
                    "abbreviated": "I.B.M.",
                },
                {
                    "original": "International Business Machines Corporation",
                    "abbreviated": "I.B.M. Corporation",
                },
            ],
        ),
        (
            "Apple Inc.",
            [],  # Single word basename, no abbreviations generated
        ),
        (
            "Johnson & Johnson Inc.",
            [
                {"original": "Johnson & Johnson Inc.", "abbreviated": "JJ"},
                {"original": "Johnson & Johnson Inc.", "abbreviated": "JJ Inc."},
                {"original": "Johnson & Johnson Inc.", "abbreviated": "J.J."},
                {"original": "Johnson & Johnson Inc.", "abbreviated": "J.J. Inc."},
            ],
        ),
    ],
)
def test_process_single_name(company_name, expected_abbreviations):
    """Test generation of company abbreviations"""
    result = process_single_name(company_name)

    assert isinstance(result, list)
    assert all(isinstance(item, dict) for item in result)
    assert all("original" in item and "abbreviated" in item for item in result)

    # Check that expected abbreviations are present
    for expected in expected_abbreviations:
        assert expected in result


def test_process_single_name_empty_input():
    """Test handling of empty and null inputs"""
    assert process_single_name(None) == []
    assert process_single_name("") == []
    assert process_single_name("ABC") == []  # No meaningful abbreviation for "ABC" alone
