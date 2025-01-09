import pickle
with open("./data/debug.pickle", "rb") as f:
    r = pickle.load(f)
print(r)
