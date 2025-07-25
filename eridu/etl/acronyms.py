import cleanco  # type: ignore
import pandas as pd
from stop_words import AVAILABLE_LANGUAGES, get_stop_words  # type: ignore


def get_multilingual_stop_words() -> set:
    """
    Get a list of ALL common stop words across languages.
    """
    stop_words: set = set()
    for lang in AVAILABLE_LANGUAGES:
        stop_words.update(get_stop_words(lang))
    return stop_words


def get_corporate_ending(company_name):
    """Extract corporate ending by finding what basename removed"""
    if not company_name:
        return ""

    cleaned = cleanco.basename(company_name)

    # If nothing was removed, there's no ending
    if cleaned == company_name:
        return ""

    # The ending is everything after the basename
    # Find where cleaned ends in the original string
    cleaned_len = len(cleaned)
    ending = company_name[cleaned_len:].strip()

    return ending if ending else ""


def process_single_name(name: str) -> list[dict[str, str]]:
    if pd.isna(name) or name is None:
        return []

    pairs: list[dict[str, str]] = []

    # Clean the company name using cleanco
    cleaned_name = cleanco.basename(name)
    ending = get_corporate_ending(name)

    # Skip if cleaning didn't change anything meaningful
    if cleaned_name and cleaned_name != name:
        # Split into words and filter
        words: list[str] = cleaned_name.split()

        # Filter out common words and single letters
        stop_words: set[str] = get_multilingual_stop_words()
        meaningful_words: list[str] = [
            w for w in words if w.lower() not in stop_words and len(w) > 1
        ]

        if meaningful_words and len(meaningful_words) > 1:  # Only process if more than one word
            # Create standard abbreviation (first letter of each word)
            abbreviation: str = "".join([w[0].upper() for w in meaningful_words])

            if len(abbreviation) > 1:  # Only add if abbreviation is meaningful
                pairs.append({"original": name, "abbreviated": abbreviation})
                if ending:
                    # Add abbreviation with suffix
                    pairs.append({"original": name, "abbreviated": f"{abbreviation} {ending}"})

                # Add dotted version
                dotted = ".".join([w[0].upper() for w in meaningful_words]) + "."
                pairs.append({"original": name, "abbreviated": dotted})
                if ending:
                    # Add dotted version with suffix
                    pairs.append({"original": name, "abbreviated": f"{dotted} {ending}"})

    return pairs
