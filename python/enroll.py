from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import argparse, os
import scipy
import cv2
import numpy as np
from cv2 import imread
from fnc.segment import segment, findTopEyelid, findBottomEyelid
from fnc.normalize import normalize
from fnc.encode import encode
from fnc.boundary import searchInnerBound, searchOuterBound
from fnc.line import findline, linecoords
from fnc.matching import calHammingDist
import multiprocessing as mp
from glob import glob
from PIL import Image


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="././CASIA1", help="Path to the directory containing CASIA1 images.")

parser.add_argument("--temp_dir", type=str, default="./templates/CASIA1", help="Path to the directory containing templates.")

#parser.add_argument("--n_cores", type=int, default=cpu_count(), help="Number of cores used for enrolling template.")

args = parser.parse_args()



# Check the existence of temp_dir
if not os.path.exists(args.temp_dir):
	print("makedirs", args.temp_dir)
	os.makedirs(args.temp_dir)

# Get list of files for enrolling template, just "xxx_1_x.jpg" files are selected
files = glob(os.path.join(args.data_dir + "/*/*_1_*.jpg"))
#files = glob(os.path.join(args.data_dir + "/*.jpg"))
print(args.data_dir)
n_files = len(files)
print("Number of files for enrolling:", n_files)

# Parallel pools to enroll templates

print("Start enrolling...")

for i in range(n_files):
    im = cv2.imread(files[i], 0)
    eyelashes_thres = 80
    # Normalisation parameters
    radial_res = 20
    angular_res = 240
    ##line 85 in segment.py had to be changed imwithnoise = imwithnoise #+ mask_top + mask_bot
    ciriris, cirpupil, imwithnoise = segment(im, eyelashes_thres, False)
    
    #cv2.imwrite(filename="im_aftersegmentation.jpg",imwithnoise)
    polar_array, noise_array = normalize(imwithnoise, ciriris[1], ciriris[0], ciriris[2],
                                            cirpupil[1], cirpupil[0], cirpupil[2],
                                            radial_res, angular_res)

    minWaveLength = 18
    mult = 1
    sigmaOnf = 0.5
    template, mask = encode(polar_array, noise_array, minWaveLength, mult, sigmaOnf)
    rowp, colp, rp = searchInnerBound(im)
    row, col, r = searchOuterBound(im, rowp, colp, rp)

    print(rowp)
    print(colp)
    print(rp)
    fig, ax = plt.subplots(1)

    circ2 = Circle((col,row), r, edgecolor = 'blue',alpha=.3)
    circ = Circle((colp,rowp), rp, edgecolor = 'red',alpha=.3)
    mask = np.zeros_like(im)
    mask = 255-mask
    imm = cv2.circle(mask, (int(colp),int(rowp)), rp, (0,0,0),-1)
    mask2 = np.zeros_like(im)
    immm = cv2.circle(mask2, (int(col),int(row)), r, (255,255,255), -1)

    mask3 = np.zeros_like(im)
    m3 = cv2.rectangle(mask3, [0,int(rowp-rp)], [320, int(rowp+rp)], (255,255,255), -1)

    newim = cv2.bitwise_and(im, imm)
    newim2 = cv2.bitwise_and(newim, immm)
    newim3 = cv2.bitwise_and(newim2, m3)

    ax.imshow(newim3, cmap="gray")
    basename = os.path.basename(files[i])
    out_file = os.path.join(args.temp_dir, (basename))
	#savemat(out_file, mdict={'template': template, 'mask': mask})
    cv2.imwrite(filename=out_file, img = newim3)
    i = i+1

