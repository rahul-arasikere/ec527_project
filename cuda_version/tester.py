import os
import subprocess
# import cv2 
# import numpy as np

result = subprocess.check_output("rm -rf thresholded *_t.png",shell=True, universal_newlines=True)
print(result)