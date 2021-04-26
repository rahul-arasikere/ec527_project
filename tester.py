import os
import subprocess
# import cv2 
# import numpy as np

# compile serial and parallel versions
subprocess.check_output("cd cuda_version && make clean && make && cd ..",shell=True, universal_newlines=True)
subprocess.check_output("cd serial_version && make clean && make && cd ..",shell=True, universal_newlines=True)

serial_times = {}
cuda_times = {}

directory = "images"
NUM_TESTS = 10

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        fname, _ = filename.split(".")
        size, ffname = fname.split("_")
        size = int(size)
        print(size)
         # print(os.path.join(directory, filename))
        serial_time_total = 0
        cuda_time_total = 0
        for i in range(NUM_TESTS):
            output = subprocess.check_output(["./serial_version/watershed", f"{directory}/{filename}"])
            serial_time_total += float(output)
            output = subprocess.check_output(["./cuda_version/watershed", f"{directory}/{filename}"])
            cuda_time_total += float(output)
        serial_times[size] = serial_time_total/NUM_TESTS
        cuda_times[size] = cuda_time_total/NUM_TESTS

    else:
        continue
subprocess.check_output("rm *.png",shell=True, universal_newlines=True)

print("serial results")
for key, value in serial_times.items():
	    print(f"{value}, {key}")
print("cuda results")
for key, value in cuda_times.items():
	    print(f"{value/1000}, {key}")

# for dirname, _, filenames in os.walk('.'):
#     for filename in filenames:
#         fname, ext = filename.split(".")
#         if(ext != "png"): continue

