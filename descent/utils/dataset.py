"""Utilities for working with datasets."""

import typing

import datasets


def iter_dataset(dataset: datasets.Dataset) -> typing.Iterator[dict[str, typing.Any]]:
    """Iterate over a Hugging Face Dataset, yielding each 'row' as a dictionary.

    Args:
        dataset: The dataset to iterate over.

    Yields:
        A dictionary representing a single entry in the batch, where each key is a
        column name and the corresponding value is the entry in that column for the
        current row.
    """

    columns = [*dataset.features]

    for row in zip(*[dataset[column] for column in columns]):
        yield {column: v for column, v in zip(columns, row)}
