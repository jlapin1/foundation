from typing import Any

import polars as pl

def listify(obj: Any) -> list[Any]:
    try:
        invalid = [str, pl.DataFrame, pl.LazyFrame]
        if any(isinstance(obj, c) for c in invalid):
            raise TypeError

        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)
