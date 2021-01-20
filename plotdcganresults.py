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

plt.plot(epochs, dLosses, "royalblue")
plt.plot(epochs, gLosses, "orange")
plt.title("DCGAN - MNIST: Custos do Gerador e do Discriminador")
plt.xlabel("Época", fontsize=fontsize)
plt.ylabel("Custo", fontsize=fontsize)
plt.xticks(fontsize=fontsize-4)
plt.yticks(fontsize=fontsize-4)
plt.legend(loc='best')
plt.savefig("dcgan-separate-losses.png")
plt.savefig("dcgan-separate-losses.pdf")
plt.clf()

plt.plot(epochs, totalLosses, "royalblue")
plt.title("DCGAN - MNIST: Custo Total")
plt.xlabel("Época", fontsize=fontsize)
plt.ylabel("Custo", fontsize=fontsize)
plt.xticks(fontsize=fontsize-4)
plt.yticks(fontsize=fontsize-4)
plt.savefig("dcgan-total-losses.png")
plt.savefig("dcgan-total-losses.pdf")
plt.clf()
