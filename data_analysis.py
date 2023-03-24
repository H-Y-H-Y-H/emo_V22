import numpy as np


list_data = np.loadtxt('select_id.csv',dtype=int)
values, counts = np.unique(list_data, return_counts=True)

print(values)
print(counts)

