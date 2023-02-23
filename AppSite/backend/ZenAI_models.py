from random import Random
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import mediapipe as mp
import time 
import traceback
import pickle

from joint_pose_vocab import vocab_dict
from ZenAI_data_transformation import combined_test, combined_train, joint_idx_map, classes, columns, feedback_classes


from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC


''' Setting current dir to the dir that hosts this python file to avoid any funny buisness'''
os.chdir(os.path.dirname(os.path.abspath(__file__)))
path = os.getcwd()

with open('models/svm_93.pkl', 'rb') as f:
    model = pickle.load(f)

def reshape_data(example):
    return pd.DataFrame(np.array(example).reshape(1, -1), columns=columns[1:])


data = reshape_data([8.390697637127042, 13.364568331384618, 16.49759248897499, 153.50000646379374, 173.20291493738577, 199.52935190007366, 179.00845878279233, 198.25172013734928])
x = model.predict(data)
print(classes[x[0]])