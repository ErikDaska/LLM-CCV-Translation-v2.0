"""Build a single evaluation example per source with N:N reference translations."""

import re
import unicodedata


def normalize_source(value):
    """Match the sentence normalization strategy of the corpus cleaner."""
    if not isinstance(value, str):
        return None
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip() or None


def build_multireference_view(dataset_split, source_column="original_sentence", target_column="translation"):
    """Return selected positional indices and a reference-list per unique source.

    Assumes the source and target columns are valid nonempty text after cleaning.
    Keeps the first row for each normalized source sentence, and distinct reference texts.
    """
    positions = []
    references = []
    grouped = {}
    for index, row in enumerate(dataset_split):
        source = normalize_source(row[source_column])
        target = row[target_column]
        if source is None or not isinstance(target, str) or not target.strip():
            raise ValueError(f"Invalid validation pair at source row {index}")
        group_idx = grouped.get(source)
        if group_idx is None:
            group_idx = len(positions)
            grouped[source] = group_idx
            positions.append(index)
            references.append([])
        target = target.strip()
        if target not in references[group_idx]:
            references[group_idx].append(target)
    return positions, tuple(tuple(refs) for refs in references)