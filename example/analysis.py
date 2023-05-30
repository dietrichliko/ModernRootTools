#

from typing import Callable
from typing import Any

import click

from mrtools import datasets
from mrtools import cache

RDataFrame = Any
CallableDefineDF = Callable[
    [RDataFrame, str, datasets.DatasetType, str], dict[str, RDataFrame]
]

DATASETS_DEF = "datasets/Met_NanoNtuple_v10_scratch.yaml"

DATASETS = "MET"

OPTIONS: dict[str, Any]


class Analysis:
    tight: bool

    def __init__(self, tight: bool):
        self.tight = tight

    def __call__(
        self,
        df: RDataFrame,
        dataset_name: str,
        dataset_type: datasets.DatasetType,
        period: str,
    ) -> dict[str, RDataFrame]:
        # df = (
        #     df.Filter()
        # }

        return {"main": df}


@click.command
@click.option("--tight/--no-tight")
def get_analysis(tight: bool) -> CallableDefineDF:
    return Analysis(tight)
