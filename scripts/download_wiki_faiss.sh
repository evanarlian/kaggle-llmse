#!/usr/bin/env bash

# https://www.kaggle.com/datasets/jjinho/wikipedia-2023-07-faiss-index
kaggle datasets download -d jjinho/wikipedia-2023-07-faiss-index -p input/wikipedia-2023-07-faiss-index
cd input/wikipedia-2023-07-faiss-index
unzip wikipedia-2023-07-faiss-index.zip
