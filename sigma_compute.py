import numpy as np

# The data as provided in the error logs
error_data = """
err: 0.003405, -0.020760, 0.411857, 0.013814, -0.008802, -0.007196
err: -0.011227, 0.028027, -0.152396, 0.020951, 0.004091, 0.000298
err: 0.004805, 0.004081, 0.019613, 0.008074, 0.001366, 0.000754
err: -0.009759, -0.000468, 0.057457, -0.002410, 0.007167, -0.000651
err: 0.003579, -0.042613, 0.432676, 0.003354, 0.000401, 0.001892
err: 0.008196, -0.010557, 0.165475, -0.001826, 0.004051, 0.004332
err: 0.008623, -0.020218, 0.020280, 0.006568, -0.001157, 0.001208
err: 0.003785, 0.005455, 0.154625, 0.011393, 0.007806, -0.003638
err: 0.026229, -0.057453, 0.503089, -0.019142, -0.007627, 0.004052
err: 0.003306, -0.016575, 0.035546, -0.000780, 0.006131, -0.002776
"""

# Parse the data
lines = [line.strip() for line in error_data.strip().split('\n')]
data = []

for line in lines:
    # Remove the 'err: ' prefix and split by comma
    values = line.replace('err: ', '').split(', ')
    # Convert to float
    values = [float(val) for val in values]
    data.append(values)

# Convert to numpy array for easier column operations
data_array = np.array(data)

# Compute standard deviation for each column
sigmas = np.std(data_array, axis=0)

# Print results
for i, sigma in enumerate(sigmas):
    print(f"Column {i+1} sigma: {sigma:.6f}")