import os
import subprocess

# compile serial and parallel versions
subprocess.check_output("cd cuda_version && make clean && make && cd ..",shell=True, universal_newlines=True)
subprocess.check_output("cd serial_version && make clean && make && cd ..",shell=True, universal_newlines=True)
subprocess.check_output("module load gcc",shell=True, universal_newlines=True)
subprocess.check_output("cd openmpi_version && make clean && make && cd ..",shell=True, universal_newlines=True)

serial_times = {}
cuda_times = {}
openmpi_times = {}

directory = "images/thresholded"
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
        openmpi_time_total = 0
        for i in range(NUM_TESTS):
            output = subprocess.check_output(["./serial_version/watershed", f"{directory}/{filename}"])
            serial_time_total += float(output)
            output = subprocess.check_output(["./cuda_version/watershed", f"{directory}/{filename}"])
            cuda_time_total += float(output)
            output = subprocess.check_output(["./openmpi_version/watershed", f"{directory}/{filename}"])
            openmpi_time_total += float(output)
        serial_times[size] = serial_time_total/NUM_TESTS
        cuda_times[size] = cuda_time_total/NUM_TESTS
        
        openmpi_times[size] = openmpi_time_total/NUM_TESTS
    else:
        continue

print("serial results")
for key, value in serial_times.items():
	    print(f"{value}, {key}")
print("cuda results")
for key, value in cuda_times.items():
	    print(f"{value/1000}, {key}")
print("openmp results")
for key, value in openmpi_times.items():
	    print(f"{value/1000}, {key}")
# for dirname, _, filenames in os.walk('.'):
#     for filename in filenames:
#         fname, ext = filename.split(".")
#         if(ext != "png"): continue

subprocess.check_output("rm *.png",shell=True, universal_newlines=True)
