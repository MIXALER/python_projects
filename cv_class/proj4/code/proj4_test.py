import cv2
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from cv_class.proj4.code.utils import *
import cv_class.proj4.code.student as sc

data_path = osp.join('..', 'data')
# Positive training examples. 36x36 head crops
train_path_pos = osp.join(data_path, 'caltech_faces', 'Caltech_CropFaces')
# Mine random or hard negatives from here
non_face_scn_path = osp.join(data_path, 'train_non_face_scenes')
# CMU+MIT test scenes
test_scn_path = osp.join(data_path, 'test_scenes', 'test_jpg')
# Ground truth face locations in the test set
label_filename = osp.join(data_path, 'test_scenes', 'ground_truth_bboxes.txt')

# The faces are 36x36 pixels, which works fine as a template size. You could
# add other fields to this dict if you want to modify HoG default
# parameters such as the number of orientations, but that does not help
# performance in our limited test.
feature_params = {'template_size': 36, 'hog_cell_size': 6}

# Number of negatives to use for training.
# Higher will work strictly better, but you should start with 10000 for debugging
num_negative_examples = 10000

features_pos = sc.get_positive_features(train_path_pos, feature_params)

features_neg = sc.get_random_negative_features(non_face_scn_path, feature_params,
                                               num_negative_examples)

svm = sc.train_classifier(features_pos, features_neg, 5e-2)

confidences = svm.decision_function(np.vstack((features_pos, features_neg)))
label_vector = np.hstack((np.ones(len(features_pos)), -np.ones(len(features_neg))))
[tp_rate, fp_rate, tn_rate, fn_rate] = report_accuracy(confidences, label_vector)

face_confs = confidences[label_vector > 0]
non_face_confs = confidences[label_vector < 0]
plt.figure()
plt.hist(np.sort(face_confs), 100, facecolor='g', histtype='step', density=1, label='faces')
plt.hist(np.sort(non_face_confs), 100, facecolor='r', histtype='step', density=1, label='non faces')
# plt.plot([0, len(non_face_confs)], [0, 0], 'b', label='decision boundary')
plt.xlabel('predicted score')
plt.ylabel('Percentage of images')
plt.legend()

visualize_hog(svm, feature_params)

hard_negs = sc.mine_hard_negs(non_face_scn_path, svm, feature_params)
features_neg_2 = np.vstack((features_neg, hard_negs))

svm_2 = sc.train_classifier(features_pos, features_neg_2, 5e-2)

bboxes, confidences, image_ids = sc.run_detector(test_scn_path, svm, feature_params)

bboxes_2, confidences_2, image_ids_2 = sc.run_detector(test_scn_path, svm_2, feature_params)

gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections = evaluate_detections(bboxes, confidences,
                                                                                    image_ids, label_filename)

gt_ids, gt_bboxes, gt_isclaimed, tp_2, fp_2, duplicate_detections_2 = evaluate_detections(bboxes_2, confidences_2,
                                                                                          image_ids_2, label_filename)

# visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_filename)
visualize_detections_by_image(bboxes_2, confidences_2, image_ids_2, tp_2, fp_2, test_scn_path, label_filename)

# test_scn_path_extra = osp.join(data_path, 'extra_test_scenes') # Bonus scenes
# bboxes_extra, confidences_extra, image_ids_extra = sc.run_detector(test_scn_path_extra, svm_2, feature_params)

# visualize_detections_by_image_no_gt(bboxes_extra, confidences_extra, image_ids_extra, test_scn_path_extra)
