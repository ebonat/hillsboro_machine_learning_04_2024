
# Calculating molecular fingerprints using padelpy
# https://dataprofessor.github.io/ds/bioinformatics/cheminformatics/padelpy/scikit-learn/2021/07/06/_07_06_padelpy.html

# PaDELPy: A Python wrapper for PaDEL-Descriptor software
# https://github.com/ECRL/PaDELPy/blob/master/README.md

import time
import os
os.system("cls")

import padelpy
from padelpy import padeldescriptor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt        
import seaborn as sns
import traceback as tb

import numpy as np
import pandas as pd
# from padelpy import from_mdl
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from imblearn.over_sampling import SMOTE
# from lazypredict.Supervised import LazyClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster import KMeans
from lazypredict.Supervised import LazyClassifier

def get_program_running(start_time):
    end_time = time.time()
    diff_time = end_time - start_time
    result = time.strftime("%H:%M:%S", time.gmtime(diff_time)) 
    print("program runtime: {}".format(result))

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

def imbalanced_classes_plot(ds_y, is_class_number, class_column_name, x_label, y_label, title, font_size=8):
    try:
        plt.figure(figsize = (5,5))
        sns.set(style="darkgrid")
        if (is_class_number == True):
            ax = sns.countplot(x=ds_y)
        else:
            ax = sns.countplot(x=class_column_name, data=ds_y)
        ax.set_xlabel(x_label,fontsize=font_size)
        ax.set_ylabel(y_label, fontsize=font_size)
        ax.tick_params(labelsize=font_size)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 10, "{:1.0f}".format(height), ha="center", fontsize=font_size)     
        plt.title(title, fontsize=font_size)
        plt.show()
    except:
        tb.print_exc()

def main():
    """hcv_ns5b_curated calculating molecular fingerprints using padelpy
    """
    print("hcv_ns5b_curated calculating molecular fingerprints using padelpy")    

    project_folder_path = r"your_folder_path"
    substructure_file_path = os.path.join(project_folder_path, "hcv_ns5b_substructure_final.csv")
    df_substructure = pd.read_csv(substructure_file_path)
    df_substructure.info()
    print(df_substructure)
    # exit()

    X = df_substructure.drop('Activity', axis=1)
    print("Before SMOTE")
    print(X.shape)

    y = df_substructure['Activity']

    smote_over_sampling = SMOTE(random_state=70, n_jobs=-1)    
    X, y = smote_over_sampling.fit_resample(X, y)
    print("After SMOTE")
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    model_classifier = RandomForestClassifier(n_jobs=-1, random_state=100)

    model_classifier.fit(X_train, y_train)

    y_predicted = model_classifier.predict(X_test) 

    # calculate classification accuracy score
    accuracy_score_value = accuracy_score(y_test, y_predicted) * 100
    accuracy_score_value = float("{0:0.2f}".format(accuracy_score_value))    
    print("classification accuracy score:")
    print(accuracy_score_value)
    print()

    # calculate classification confusion matrix
    confusion_matrix_result = confusion_matrix(y_test, y_predicted)
    print("classification confusion matrix:")
    print(confusion_matrix_result)
    print()

    # calculate classification report
    classification_report_result = classification_report(y_test,y_predicted)
    print("classification report:")    
    print(classification_report_result)
    print()  

if __name__ == '__main__':
    start_time = time.time()    
    main()
    get_program_running(start_time)
