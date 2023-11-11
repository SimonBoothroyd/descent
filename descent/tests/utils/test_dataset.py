import datasets

from descent.utils.dataset import iter_dataset


def test_iter_dataset():
    test_dataset = datasets.Dataset.from_dict(
        {"column1": ["data1", "data2", "data3"], "column2": ["data4", "data5", "data6"]}
    )
    columns = ["column1", "column2"]

    expected_output = [
        {"column1": "data1", "column2": "data4"},
        {"column1": "data2", "column2": "data5"},
        {"column1": "data3", "column2": "data6"},
    ]

    output = list(iter_dataset(test_dataset, columns))

    assert output == expected_output
