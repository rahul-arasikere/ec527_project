import os
import subprocess
import cv2 
import numpy as np

subprocess.check_output("rm -rf thresholded *_t.png",shell=True, universal_newlines=True)

for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        fname, ext = filename.split(".")
        if(ext != "png"): continue
        print(filename)
        image1 = cv2.imread(filename) 
        img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 127, 5)
        cv2.imwrite(f"{fname}_t.png", thresh)

subprocess.check_output(f"mkdir thresholded && mv *_t.png thresholded",shell=True, universal_newlines=True)
