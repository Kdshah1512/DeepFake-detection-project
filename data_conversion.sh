#!/bin/bash

if [ ! -f ".project-root" ]; then
  echo "Please run this script from the root of the project"
  exit 1
fi

DATASET_DIR=datasets/CDFv2
CONFIG_DIR=config/datasets/CDFv2

mkdir -p $DATASET_DIR
mkdir -p $CONFIG_DIR/test

find $DATASET_DIR/Celeb-real/* -type f | sort > $CONFIG_DIR/test/Celeb-real.txt
find $DATASET_DIR/Celeb-synthesis/* -type f | sort > $CONFIG_DIR/test/Celeb-synthesis.txt
find $DATASET_DIR/YouTube-real/* -type f | sort > $CONFIG_DIR/test/YouTube-real.txt
# find $DATASET_DIR/NT/* -type f | sort > $CONFIG_DIR/test/NT.txt
# find $DATASET_DIR/real/* -type f | sort > $CONFIG_DIR/test/real.txt