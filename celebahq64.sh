#!/bin/bash
for set in val train; do
  mkdir -p data/celeba_hq_64/$set/1
  for img in $(ls data/celeba_hq/$set | grep .jpg); do
    convert data/celeba_hq/$set/$img -resize 64x64 data/celeba_hq_64/$set/1/$img &
  done
done
