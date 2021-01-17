#!/bin/bash
for set in val train; do
  mkdir -p data/celeba_hq_128/$set
  for img in $(ls data/celeba_hq/$set | grep .jpg); do
    convert data/celeba_hq/$set/$img -resize 128x128 data/celeba_hq_128/$set/$img &
  done
done
