import csv
import matplotlib.pyplot as plt
import numpy as np

epochs = []
dLosses = []
gLosses = []
totalLosses = []

with open("results.csv", 'r') as resultsFile:
  reader = csv.DictReader(resultsFile)
  
  for row in reader:
    epochs.append(int(row["Epoch"]))
    dLosses.append(float(row["DLoss"]))
    gLosses.append(float(row["GLoss"]))
    totalLosses.append(float(row["TotalLoss"]))

plt.locator_params(nbins=5)
plt.plot(epochs, dLosses, "royalblue", label="Discriminador")
plt.plot(epochs, gLosses, "orange", label="Gerador")
plt.xticks(np.arange(0, len(epochs), 100))
plt.yticks(np.arange(0, max(max(dLosses), max(gLosses)), 2))
plt.title("DCGAN - MNIST: Custos do Gerador e do Discriminador")
plt.xlabel("Época")
plt.ylabel("Custo")
plt.legend(loc='best')
plt.savefig("dcgan-separate-losses.png")
plt.savefig("dcgan-separate-losses.pdf")
plt.clf()

plt.plot(epochs, totalLosses, "royalblue")
plt.xticks(np.arange(0, len(epochs), 100))
plt.yticks(np.arange(0, max(totalLosses), 2))
plt.title("DCGAN - MNIST: Custo Total")
plt.xlabel("Época")
plt.ylabel("Custo")
plt.savefig("dcgan-total-losses.png")
plt.savefig("dcgan-total-losses.pdf")
plt.clf()
