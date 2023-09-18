import pandas as pd


def wider(df: pd.DataFrame):
    with pd.option_context("display.max_colwidth", None):
        display(df)
