#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import os
import numpy as np
from glob import glob
#from tqdm import tqdm
from time import time
from random import shuffle
from matplotlib import pyplot as plt
from itertools import repeat
from collections import defaultdict
#from multiprocessing import Pool, cpu_count

from fnc.extractFeature import extractFeature
from fnc.matching import calHammingDist


#------------------------------------------------------------------------------
#   Parameters
#------------------------------------------------------------------------------
CASIA1_DIR = "/Users/overbydl/Iris-Recognition/CASIA1"
EYELASHES_THRES = 80
N_IMAGES = 4


#------------------------------------------------------------------------------
#   Pool function of extracting feature
#------------------------------------------------------------------------------
def pool_func_extract_feature(args):
    im_filename, eyelashes_thres, use_multiprocess = args

    template, mask, im_filename = extractFeature(
        im_filename=im_filename,
        eyelashes_thres=eyelashes_thres,
        use_multiprocess=False,
    )
    return template, mask, im_filename


#------------------------------------------------------------------------------
#   Pool function of calculating Hamming distance
#------------------------------------------------------------------------------
def pool_func_calHammingDist(args):
    template1, mask1, template2, mask2 = args
    dist = calHammingDist(template1, mask1, template2, mask2)
    return dist


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Get identities of MMU2 dataset
'''''
identities = glob(os.path.join(CASIA1_DIR, "**"))
identities = sorted([os.path.basename(identity) for identity in identities])
n_identities = len(identities)
print("Number of identities:", n_identities)


# Construct a dictionary of files
files_dict = {}
image_files = []
#for identity in identities:
files = glob(os.path.join(CASIA1_DIR + "*.*")) #add /*2*/ for specific directories
    #shuffle(files)
files_dict[0] = files[:N_IMAGES]
    # print("Identity %s: %d images" % (identity, len(files_dict[identity])))
image_files += files[:N_IMAGES]

n_image_files = len(image_files)
print("Number of image files:", n_image_files)


features = []
start_time = time()
for i in range(n_image_files):
    args = (image_files[i], EYELASHES_THRES, False)
    features.append(pool_func_extract_feature(args))
    i = i+1
finish_time = time()
print("Extraction time: %.3f [s]" % (finish_time-start_time))


# Calculate the distances
args = []
distances = []
start_time = time()
for i in range(n_image_files):
    for j in range(n_image_files):
        if i>=j:
            continue
        
        arg = (features[i][0], features[i][1], features[j][0], features[j][1])
        args.append(arg)
        distances.append(pool_func_calHammingDist(arg))
print("Number of pairs:", len(args))
finish_time = time()

print("Extraction time: %.3f [s]" % (finish_time-start_time))


# Construct a distance matrix
dist_mat = np.zeros([n_image_files, n_image_files])
k = 0
for i in range(n_image_files):
    for j in range(n_image_files):
        if i<j:
            dist_mat[i, j] = distances[k]
            k += 1
        elif i>j:
            dist_mat[i, j] = dist_mat[j, i]
np.savetxt("dist_mat_csv_casia.csv", dist_mat, delimiter=',')


np.save("dist_mat_casia1.npy", dist_mat)

plt.figure()
plt.imshow(dist_mat)
plt.show()
'''''
identities = glob(os.path.join(CASIA1_DIR, "**"))
identities = sorted([os.path.basename(identity) for identity in identities])
n_identities = len(identities)
print("Number of identities:", n_identities)


# Construct a dictionary of files
files_dict = {}
image_files = []
for identity in identities:
    files = glob(os.path.join(CASIA1_DIR, identity, "*.*"))
    #shuffle(files)
    files_dict[identity] = files[:N_IMAGES]
    # print("Identity %s: %d images" % (identity, len(files_dict[identity])))
    image_files += files[:N_IMAGES]

n_image_files = len(image_files)
print("Number of image files:", n_image_files)


features = []
start_time = time()
for i in range(n_image_files):
    args = (image_files[i], EYELASHES_THRES, False)
    features.append(pool_func_extract_feature(args))
    i = i+1
finish_time = time()
print("Extraction time: %.3f [s]" % (finish_time-start_time))


# Calculate the distances
args = []
distances = []
start_time = time()
for i in range(n_image_files):
    for j in range(n_image_files):
        if i>=j:
            continue
        arg = (features[i][0], features[i][1], features[j][0], features[j][1])
        print(arg)
        args.append(arg)
        distances.append(pool_func_calHammingDist(arg))
print("Number of pairs:", len(args))
finish_time = time()

print("Extraction time: %.3f [s]" % (finish_time-start_time))


# Construct a distance matrix
dist_mat = np.zeros([n_image_files, n_image_files])
k = 0
for i in range(n_image_files):
    for j in range(n_image_files):
        if i<j:
            dist_mat[i, j] = distances[k]
            k += 1
        elif i>j:
            dist_mat[i, j] = dist_mat[j, i]

np.savetxt("dist_mat_csv_casia.csv", dist_mat, delimiter=',')
for i in range(n_image_files):
    print(image_files[i])
np.save("dist_mat_casia1.npy", dist_mat)

plt.figure()
plt.imshow(dist_mat)
plt.show()

'''''
# Extract features
features = []
start_time = time()
for i in range(n_image_files):
    args = (image_files[i], EYELASHES_THRES, False)
    features.append(pool_func_extract_feature(args))
    i = i+1
finish_time = time()
print("Extraction time: %.3f [s]" % (finish_time-start_time))


# Calculate the distances
args = []
distances = []
start_time = time()
for i in range(n_image_files):
    for j in range(n_image_files):
        if i>=j:
            continue
        
        arg = (features[i][0], features[i][1], features[j][0], features[j][1])
        args.append(arg)
        distances.append(pool_func_calHammingDist(arg))
print("Number of pairs:", len(args))
finish_time = time()

print("Extraction time: %.3f [s]" % (finish_time-start_time))


# Construct a distance matrix
dist_mat = np.zeros([n_image_files, n_image_files])
k = 0
for i in range(n_image_files):
    for j in range(n_image_files):
        if i<j:
            dist_mat[i, j] = distances[k]
            k += 1
        elif i>j:
            dist_mat[i, j] = dist_mat[j, i]

np.save("dist_mat_casia1.npy", dist_mat)

plt.figure()
plt.imshow(dist_mat)
plt.show()
'''