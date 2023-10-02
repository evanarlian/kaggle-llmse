#!/usr/bin/env bash

# https://www.kaggle.com/datasets/yalickj/dataset-wiki-new-1
kaggle datasets download -d yalickj/dataset-wiki-new-1 -p input/dataset-wiki-new-1
cd input/dataset-wiki-new-1
unzip dataset-wiki-new-1.zip
