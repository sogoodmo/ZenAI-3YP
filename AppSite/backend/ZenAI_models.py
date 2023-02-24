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
from ZenAI_data_transformation import generate_data
from ZenAI_ideal_angles import calculate_ideal_angles


from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC


''' Setting current dir to the dir that hosts this python file to avoid any funny buisness'''


"""
    Generic type for a model of this purpose. All that is required is the models dump via a pkl file (or any that works)
"""
class Model:
    def __init__(self, columns, classes):
        self.columns = columns 
        self.classes = classes 
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def reshape_data(self, example):
        # Reshape the input data so that it can be fed into the classifier
        return pd.DataFrame(np.array(example).reshape(1, -1), columns=self.columns[1:])
    
    def predict(self, example, clf):
        # Predict the output for the input example using the given classifier
        formatted_example = self.reshape_data(example)
        probabilty_classes = clf.predict_proba(formatted_example)

        # Convert the predicted probabilities to a list of tuples containing the class and probability
        prob_predicted_classes = [(self.classes[i], p) for i, p in enumerate(probabilty_classes[0])]
        
        # Get the highest predicted class 
        predicted_class, prob = max(prob_predicted_classes, key=lambda x: x[1])
        
        return predicted_class, prob, sorted(prob_predicted_classes, key=lambda x: x[1], reverse=True)


class ZenRandomForest(Model):
    def __init__(self, columns, classes):
        super().__init__(columns, classes)

        with open('models/random_forest_92.pkl', 'rb') as f:
            self.forest = pickle.load(f)

    def predict(self, example):
        # Predict the output for the input example using the random forest classifier
        return super().predict(example, self.forest)


class ZenKNN(Model):
    def __init__(self, columns, classes):
        super().__init__(columns, classes)

        with open('models/knn_93.pkl', 'rb') as f:
            self.knn = pickle.load(f)

    def predict(self, example):
        # Predict the output for the input example using the k-nearest neighbors classifier
        return super().predict(example, self.knn)


class ZenNN(Model):
    def __init__(self, columns, classes):
        super().__init__(columns, classes)

        with open('models/nn_93.pkl', 'rb') as f:
            self.nn = pickle.load(f)

    def predict(self, example):
        # Predict the output for the input example using the neural network classifier
        return super().predict(example, self.nn)


class ZenSVM(Model):
    def __init__(self, columns, classes):
        super().__init__(columns, classes)
        with open('models/svm_93.pkl', 'rb') as f:
            self.svm = pickle.load(f)

    def predict(self, example):
        # Predict the output for the input example using the support vector machine classifier
        return super().predict(example, self.svm)