import numpy as np
import os
import glob

print("da")

file_path = r"pythonProaject1\testFFTResized\Fetal abdomen\Patient00168_Plane2_2_of_2.png.npy"
print("File exists:", os.path.exists(file_path))

print("Current working directory:", os.getcwd())

files = glob.glob("pythonProject1/testFFTResized/Fetal abdomen/*.npy")
print(files)

image = np.load("pythonProject1/testFFTResized/Fetal abdomen/Patient00168_Plane2_2_of_2.png.npy")
print(image.shape)