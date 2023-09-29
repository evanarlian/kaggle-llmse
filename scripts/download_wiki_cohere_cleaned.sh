#!/usr/bin/env bash

# https://www.kaggle.com/datasets/mbanaei/all-paraphs-parsed-expanded
kaggle datasets download -d mbanaei/all-paraphs-parsed-expanded -p input/all-paraphs-parsed-expanded
cd input/all-paraphs-parsed-expanded
unzip all-paraphs-parsed-expanded.zip
