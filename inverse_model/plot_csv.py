import matplotlib.pyplot as plt
import numpy as np

log_path = "log_MSE/"

training_L = np.loadtxt(log_path + "training_MSE.csv")
testing_L = np.loadtxt(log_path + "testing_MSE.csv")

plt.plot([i for i in range(len(training_L))], training_L, label="training")
plt.plot([i for i in range(len(training_L))], testing_L, label="testing")
plt.legend()
plt.title("MSE loss, best=0.0139")
plt.savefig(log_path+"MSE_plot.png")
plt.show()