import pandas as pd
import numpy as np

# Block
vals = [[62.5, 58.64, 50.16, 54.2],
        [33.12, 46.94, 50.13, 40.22],
        [44.8, 47.15, 47.63, 39.36],
        [52.61, 62.73, 49.12, 53.48]]

labels = ['AlgAATer', 'BBL', 'EEE', 'S++']

matrix = pd.DataFrame(np.array(vals), columns=labels)

print(matrix)

# Block user study
vals = [[25, 22.04, 21.42, 20.03, 20.24, 14.1, 18.87],
        [15.22, 18.62, 21.6, 16.17, 18.79, 14.06, 19.56],
        [16, 17.81, 17.99, 14.18, 15.32, 9.31, 13.63],
        [18.17, 20.52, 18.96, 18.89, 18.08, 12.43, 18.06],
        [12.28, 12.94, 16.85, 8.41, np.nan, np.nan, np.nan],
        [13.82, 16.42, 18.35, 10.96, np.nan, np.nan, np.nan],
        [12.93, 17.83, 16.78, 14.65, np.nan, np.nan, np.nan]]

labels = ['AlgAATer', 'BBL', 'EEE', 'S++', 'Human-Regular', 'Human-Bully', 'Human-Coop']

matrix = pd.DataFrame(np.array(vals), columns=labels)

print(matrix)

# Chicken
vals = [[50.0, 48.05, -19.48, -10.22],
        [33.68, 17.76, -33.62, -30.13],
        [67.12, 102.22, -32.54, -21.05],
        [72.79, 103.19, -0.26, -24.05]]

labels = ['AlgAATer', 'BBL', 'EEE', 'S++']

matrix = pd.DataFrame(np.array(vals), columns=labels)

print(matrix)

# Coordination
vals = [[100, 75.93, 84.95, 96.16],
        [75.93, 74.71, 72.27, 76.89],
        [84.95, 72.27, 74.9, 82.48],
        [96.16, 76.89, 82.48, 94.81]]

labels = ['AlgAATer', 'BBL', 'EEE', 'S++']

matrix = pd.DataFrame(np.array(vals), columns=labels)

print(matrix)

# Pennies
vals = [[0, -6.7, 5.02, 21.36],
        [6.7, 0, 10.03, 18.56],
        [-5.02, -10.03, 0, 5.98],
        [-21.36, -18.56, -5.98, 0]]

labels = ['AlgAATer', 'BBL', 'EEE', 'S++']

matrix = pd.DataFrame(np.array(vals), columns=labels)

print(matrix)

# Prisoners
vals = [[150, 29.12, 37.4, 127.12],
        [43.84, -12.43, 26.82, 116.04],
        [7.53, -53.29, 4.93, 109.02],
        [66.74, -57.05, -9.13, 117.02]]

labels = ['AlgAATer', 'BBL', 'EEE', 'S++']

matrix = pd.DataFrame(np.array(vals), columns=labels)

print(matrix)
