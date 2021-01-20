import csv
import matplotlib.pyplot as plt

epochs = []
dLosses = []
gLosses = []
totalLosses = []

with open("results.csv", 'r') as resultsFile:
  reader = csv.DictReader(resultsFile)
  
  for row in reader:
    epochs.append(row["Epoch"])
    dLosses.append(row["DLoss"])
    gLosses.append(row["GLoss"])
    totalLosses.append(row["TotalLoss"])

plt.locator_params(nbins=5)
plt.plot(epochs, dLosses, "royalblue", label="Discriminador")
plt.plot(epochs, gLosses, "orange", label="Gerador")
plt.title("DCGAN - MNIST: Custos do Gerador e do Discriminador")
plt.xlabel("Época")
plt.ylabel("Custo")
plt.legend(loc='best')
plt.savefig("dcgan-separate-losses.png")
plt.savefig("dcgan-separate-losses.pdf")
plt.clf()

plt.plot(epochs, totalLosses, "royalblue")
plt.title("DCGAN - MNIST: Custo Total")
plt.xlabel("Época")
plt.ylabel("Custo")
plt.savefig("dcgan-total-losses.png")
plt.savefig("dcgan-total-losses.pdf")
plt.clf()
