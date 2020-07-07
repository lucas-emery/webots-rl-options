import pickle
import matplotlib.pyplot as plt
from agents import tabular_agent
import sys

# sys.modules['tabular_agent'] = tabular_agent

with open("../pickles/data.p", "rb") as f:
    data = pickle.load(f)
    history = data['history']

plt.plot(history)
plt.show()

