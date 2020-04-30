from implementations import *
from proj1_helpers import *
import numpy as np 
'''ONLY FOR VISUALIZATION'''
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
'''ONLY FOR VISUALIZATION'''
import timeit


'''PLEASE MODIFY DATA PATHS FOR BOTH TRAIN AND TEST'''
#DATA_TRAIN_PATH = '../../data_project1/train.csv'
DATA_TRAIN_PATH = str(input("Please enter full path for training data(pwd):"))
y, tX_old, ids = load_csv_data(DATA_TRAIN_PATH)

#DATA_TEST_PATH = '../../data_project1/test.csv' 
DATA_TEST_PATH = str(input("Please enter full path for testing data(pwd):"))
_, tX_test_old, ids_test = load_csv_data(DATA_TEST_PATH)


'''PLEASE REMOVE DATA COMMENTS IF YOU WANT TO WORK WITH OTHER LEARNING ALGORITHMS'''
#(y_cat, tX_cat, ids_cat, ind_cat), (y_val_cat, tX_val_cat, ids_val_cat, ind_val_cat) = BuildDataModel_Train(y,tX_old,ids)

y_cv_cat, tX_cv_cat, ids_cv_cat, ind_cv_cat = BuildDataModel_CV(y,tX_old,ids)

#tX_test_cat, id_test_cat, ind_test_cat = BuildDataModel_Test(tX_test_old,ids_test)

tX_cv_test_cat, id_cv_test_cat, ind_cv_test_cat = BuildDataModel_CV_Test(tX_test_old,ids_test)

'''PLEASE UNCOMMENT BELOW IF YOU WANT TO SEE OTHER LEARNING ALGORITHM RESULTS'''
#Randomized Data Split
#w1 = Main(y_cat, tX_cat, y_val_cat, tX_val_cat, y, tX_old, ids, 1)
#w2 = Main(y_cat, tX_cat, y_val_cat, tX_val_cat, y, tX_old, ids, 2)
#w3 = Main(y_cat, tX_cat, y_val_cat, tX_val_cat, y, tX_old, ids, 3)
#w4 = Main(y_cat, tX_cat, y_val_cat, tX_val_cat, y, tX_old, ids, 4)
#w5 = Main(y_cat, tX_cat, y_val_cat, tX_val_cat, y, tX_old, ids, 5)
#w6 = Main(y_cat, tX_cat, y_val_cat, tX_val_cat, y, tX_old, ids, 6)

w_cv = CV_Main(y_cv_cat, tX_cv_cat)

'''PLEASE UNCOMMENT BELOW IF YOU WANT TO CREATE PREDICTIONS FOR OTHER LEARNING ALGORITHMS'''
#Results

#Tester(w1, tX_test_cat, id_test_cat, ind_test_cat,1)
#Tester(w2, tX_test_cat, id_test_cat, ind_test_cat,2)
#Tester(w3, tX_cv_test_cat, id_cv_test_cat, ind_cv_test_cat,3)
#Tester(w4, tX_cv_test_cat, id_cv_test_cat, ind_cv_test_cat,4)
#Tester(w5, tX_test_cat, id_test_cat, ind_test_cat,5)
#Tester(w6, tX_test_cat, id_test_cat, ind_test_cat,6)

Tester(w_cv, tX_cv_test_cat, id_cv_test_cat, ind_cv_test_cat,7)