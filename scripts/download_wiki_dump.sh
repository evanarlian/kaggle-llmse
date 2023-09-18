#!/usr/bin/env bash

# https://www.kaggle.com/datasets/jjinho/wikipedia-20230701
kaggle datasets download -d jjinho/wikipedia-20230701 -p input/wikipedia-20230701
cd input/wikipedia-20230701
unzip wikipedia-20230701.zip
