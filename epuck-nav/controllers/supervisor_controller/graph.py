import pickle
import matplotlib.pyplot as plt

with open("data.p", "rb") as f:
    data = pickle.load(f)
    history = data['history']

plt.plot(history)
plt.show()

