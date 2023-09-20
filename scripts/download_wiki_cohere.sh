#!/usr/bin/env bash

# https://www.kaggle.com/datasets/mbanaei/stem-wiki-cohere-no-emb
kaggle datasets download -d mbanaei/stem-wiki-cohere-no-emb -p input/stem-wiki-cohere-no-emb
cd input/stem-wiki-cohere-no-emb
unzip stem-wiki-cohere-no-emb.zip
