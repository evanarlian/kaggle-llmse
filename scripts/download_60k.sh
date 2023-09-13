#!/usr/bin/env bash

# https://www.kaggle.com/datasets/cdeotte/60k-data-with-context-v2
kaggle datasets download -d cdeotte/60k-data-with-context-v2 -p input/60k-data-with-context-v2
cd input/60k-data-with-context-v2
unzip 60k-data-with-context-v2.zip
