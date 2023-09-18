import pandas as pd
from IPython.display import display


def wider(df: pd.DataFrame):
    with pd.option_context("display.max_colwidth", None):
        display(df)
