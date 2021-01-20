#!/bin/bash

echo "Epoch, DLoss, GLoss, TotalLoss" > results.csv
cat logdcgan.txt | grep "Batch 937/938" | awk '{print $2}' | awk -F/ '{print $1}' > epoch.txt
cat logdcgan.txt | grep "Batch 937/938" | awk '{print $7}' | awk -F] '{print $1}' > dlosses.txt
cat logdcgan.txt | grep "Batch 937/938" | awk '{print $10}' | awk -F] '{print $1}' > glosses.txt
cat logdcgan.txt | grep "Batch 937/938" | awk '{print $13}' | awk -F] '{print $1}' > totallosses.txt
paste -d ',' epoch.txt dlosses.txt glosses.txt totallosses.txt >> results.csv
rm epoch.txt dlosses.txt glosses.txt totallosses.txt

