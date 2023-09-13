#!/usr/bin/env bash

# https://www.kaggle.com/datasets/nbroad/wiki-20220301-en-sci
kaggle datasets download -d nbroad/wiki-20220301-en-sci -p input/wiki-20220301-en-sci
cd input/wiki-20220301-en-sci
unzip wiki-20220301-en-sci.zip
