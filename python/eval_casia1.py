#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import os
import numpy as np
from glob import glob
#from tqdm import tqdm
from random import shuffle
from itertools import repeat
from collections import defaultdict
#from multiprocessing import Pool, cpu_count

from fnc.extractFeature import extractFeature
from fnc.matching import calHammingDist


#------------------------------------------------------------------------------
#   Parameters
#------------------------------------------------------------------------------
CASIA1_DIR = "../CASIA1"
N_IMAGES = 4

eyelashes_thresholds = np.linspace(start=10, stop=250, num=25)
thresholds = np.linspace(start=0.0, stop=1.0, num=100)


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
identities = glob(os.path.join(CASIA1_DIR, "**"))
identities = sorted([os.path.basename(identity) for identity in identities])
n_identities = len(identities)
print("Number of identities:", n_identities)


# Construct a dictionary of files
files_dict = {}
image_files = []
for identity in identities:
    files = glob(os.path.join(CASIA1_DIR, identity, "*.*"))
    shuffle(files)
    files_dict[identity] = files[:N_IMAGES]
    image_files += files[:N_IMAGES]

n_image_files = len(image_files)
print("Number of image files:", n_image_files)


# Ground truth
ground_truth = np.zeros([n_image_files, n_image_files], dtype=int)
for i in range(ground_truth.shape[0]):
    for j in range(ground_truth.shape[1]):
        if i//N_IMAGES == j//N_IMAGES:
            ground_truth[i, j] = 1
'''''
best_results = []
accuracies = []
precisions = []
recalls = []
fscores = []
thres = 0
iter = 0
for eye_threshold in (eyelashes_thresholds, range(len(eyelashes_thresholds))):
    print('thres', eye_threshold)
    for ii in range(7):
        features = []
        # Extract features
        for count in range(1):
            args = (image_files[ii], eye_threshold[thres], False)
            pfef = pool_func_extract_feature(args)
            features.append(pfef)
            #print(image_files[count])
            #count = count+1
            args = (image_files[ii+1], eye_threshold[thres], False)
            pfef = pool_func_extract_feature(args)
            features.append(pfef)
        thres = thres+1
        args = []
        distances = []
        for i in range(2):
            for j in range(2):
                if i>=j:
                    continue
                arg = (features[i][0], features[i][1], features[j][0], features[j][1])
                args.append(arg)
                #print(arg)
                distances.append(pool_func_calHammingDist(arg))
        

        # Construct a distance matrix
        k = 0
        dist_mat = np.zeros([2, 2])
        for i in range(2):
            for j in range(2):
                if i<j:
                    dist_mat[i, j] = distances[k]
                    k += 1
                elif i>j:
                    dist_mat[i, j] = dist_mat[j, i]
        print("dist comparing", ii, "to", (ii + 1), distances, 'iteration', iter)

        ii = ii + 1
        
        for threshold in thresholds:
            decision_map = (dist_mat<=threshold).astype(int)
            accuracy = (decision_map==ground_truth).sum() / ground_truth.size
            precision = (ground_truth*decision_map).sum() / decision_map.sum()
            recall = (ground_truth*decision_map).sum() / ground_truth.sum()
            fscore = 2*precision*recall / (precision+recall)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)
    # Save the best result
    best_fscore = max(fscores)
    best_threshold = thresholds[fscores.index(best_fscore)]
    best_results.append((eye_threshold[thres], best_threshold, best_fscore))

# Show the final best result
eye_thresholds = [item[0] for item in best_results]
thresholds = [item[1] for item in best_results]
fscores = [item[2] for item in best_results]

print("Maximum fscore:", max(fscores))
print("Best eye_threshold:", eye_thresholds[fscores.index(max(fscores))])
print("Best threshold:", thresholds[fscores.index(max(fscores))])
# Evaluate parameters
'''

best_results = []
accuracies = []
precisions = []
recalls = []
fscores = []
thres = 0
for eye_threshold in (eyelashes_thresholds, range(len(eyelashes_thresholds))):

    features = []
    # Extract features
    for count in range(n_image_files):
        args = (image_files[count], eye_threshold[thres], False)
        pfef = pool_func_extract_feature(args)
        features.append(pfef)
        count = count+1
    thres = thres+1
    args = []
    distances = []
    for i in range(n_image_files):
        for j in range(n_image_files):
            if i>=j:
                continue
            arg = (features[i][0], features[i][1], features[j][0], features[j][1])
            args.append(arg)
            print(arg)
            distances.append(pool_func_calHammingDist(arg))
    

    # Construct a distance matrix
    k = 0
    dist_mat = np.zeros([n_image_files, n_image_files])
    for i in range(n_image_files):
        for j in range(n_image_files):
            if i<j:
                dist_mat[i, j] = distances[k]
                k += 1
            elif i>j:
                dist_mat[i, j] = dist_mat[j, i]


    
    for threshold in thresholds:
        decision_map = (dist_mat<=threshold).astype(int)
        accuracy = (decision_map==ground_truth).sum() / ground_truth.size
        precision = (ground_truth*decision_map).sum() / decision_map.sum()
        recall = (ground_truth*decision_map).sum() / ground_truth.sum()
        fscore = 2*precision*recall / (precision+recall)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)

    # Save the best result
    best_fscore = max(fscores)
    best_threshold = thresholds[fscores.index(best_fscore)]
    best_results.append((eye_threshold[thres], best_threshold, best_fscore))

# Show the final best result
eye_thresholds = [item[0] for item in best_results]
thresholds = [item[1] for item in best_results]
fscores = [item[2] for item in best_results]

print("Maximum fscore:", max(fscores))
print("Best eye_threshold:", eye_thresholds[fscores.index(max(fscores))])
print("Best threshold:", thresholds[fscores.index(max(fscores))])


