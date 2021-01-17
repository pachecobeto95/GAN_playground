#!/bin/bash
for set in val train; do
  for gender in male female; do
  mkdir -p data/celeba_hq_128/$set/$gender
    for img in $(ls data/celeba_hq/$set/$gender | grep .jpg); do
      convert data/celeba_hq/$set/$gender/$img -resize 128x128 data/celeba_hq_128/$set/$gender/$img &
    done
  done
done
