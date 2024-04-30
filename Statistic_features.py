import numpy as np
from scipy.io import loadmat
import os

folder_path =r'C:\Users\awsom\Documents\GitHub\feature_extraction\mixed' 

file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mat')]


data = []

# Load each .mat file
for file_path in file_paths:
    mat_data = loadmat(file_path)
    data.append(mat_data)

gyroThumbX = [trial['gyroThumbX'] for trial in data]
gyroThumbY = [trial['gyroThumbY'] for trial in data]

gyroIndexX = [trial['gyroIndexX'] for trial in data]
gyroIndexY = [trial['gyroIndexY'] for trial in data]

diagnosis = [trial['diagnosis'][0] for trial in data]
min_size = min(len(trial['gyroThumbX'][0]) for trial in data)
gyroThumbX_trimmed = [arr[:min_size, :min_size] for arr in gyroThumbX]
gyroThumbY_trimmed = [arr[:min_size, :min_size] for arr in gyroThumbY]
gyroIndexX_trimmed = [arr[:min_size, :min_size] for arr in gyroIndexX]
gyroIndexY_trimmed = [arr[:min_size, :min_size] for arr in gyroIndexY]

print(min_size)
kinematic_features = []

for i in range(len(gyroThumbX_trimmed)):
    # Combine x, y, z coordinates for thumb and index gyroscopes
    thumb_xy = np.stack((gyroThumbX_trimmed[i], gyroThumbY_trimmed[i]), axis=-1)
    index_xy = np.stack((gyroIndexX_trimmed[i], gyroIndexY_trimmed[i]), axis=-1)
    thumb_diff = np.diff(thumb_xy, axis=1)
    index_diff = np.diff(index_xy, axis=1)
    thumb_velocity = np.mean(np.linalg.norm(thumb_diff, axis=-1)) if len(thumb_diff) > 0 else np.nan
    index_velocity = np.mean(np.linalg.norm(index_diff, axis=-1)) if len(index_diff) > 0 else np.nan
    
    # Compute accelerations for thumb and index gyroscopes
    thumb_acceleration = np.mean(np.linalg.norm(np.diff(thumb_diff, axis=1), axis=-1))
    index_acceleration = np.mean(np.linalg.norm(np.diff(index_diff, axis=1), axis=-1)) 
    
    thumb_jerk = np.mean(np.linalg.norm(np.diff(thumb_diff, axis=1, n=2), axis=-1)) 
    index_jerk = np.mean(np.linalg.norm(np.diff(index_diff, axis=1, n=2), axis=-1)) 
    
    # Append computed features to the list
    kinematic_features.append([thumb_velocity, thumb_acceleration, thumb_jerk,
                               index_velocity, index_acceleration, index_jerk])
# Convert list to numpy array
kinematic_features = np.array(kinematic_features)

# Print kinematic features
print("Kinematic Features:")
print(kinematic_features)
output_file = "kinematic_features.txt"
np.savetxt(output_file, kinematic_features)