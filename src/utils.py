import re

import pandas as pd
from IPython.display import display


def wider(df: pd.DataFrame):
    with pd.option_context("display.max_colwidth", None):
        display(df)


def dirx(obj, pat=None):
    """dir() extended, can filter by regex."""
    members = dir(obj)
    if pat is not None:
        members = [m for m in members if re.search(pat, m)]
    return members
