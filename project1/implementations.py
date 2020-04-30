######################## DATA ENGINEERING ################################
##########################################################################

import numpy as np
'''ONLY FOR VISUALIZATION'''
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
'''ONLY FOR VISUALIZATION'''
from proj1_helpers import *
import timeit

'''DATASET INTRINSICS AND SHAPE (TARGETS AND IDS INCLUDED)'''
def DataSetInfo(y, tX_old, ids):
    print("Training examples: ", tX_old, " & shape: ")
    print("Targets: ", y)
    print("Ids: ",ids)
    print("Shapes of tX, y & Ids: ", tX_old.shape, y.shape, ids.shape)

'''INITIALIZE WEIGHTS WITH RANDOM VALUES OR ZEROS'''
def InitWeights(feat):
    #ww = np.random.rand(feat)
    ww = np.zeros((feat))
    init_w = np.array(ww, dtype=np.float64)
    return init_w

'''HYPER PARAMETERS'''
'''THESE PARAMETERS ARE CHOSEN ACCORDING TO TUNING AND TESTING'''
def HyperParameters():
    max_iter = 500
    epochs = 5
    gamma = 1e-2
    lambda_ = 1e-1
    decay = 1e-2
    k_fold = 10
    return max_iter, epochs, gamma, lambda_, decay, k_fold

'''DECREASE LEARNING RATE AT EVERY EPOCH'''
def GammaScheduler(gamma, decay, epoch):
    '''TIME BASED DECAY METHOD'''
    gamma = gamma * (1/(1 + decay * epoch))
    return gamma

'''TAKE LOG TRANSFORMATION OF FEATURES'''
def LogTransformData(tX, features):  
    data = tX[:, features]
    indices = np.where(data > -999)
    data[indices] = np.log(1 + data[indices])
    tX = ManipulateFeatures(tX, data, features)    
    return tX

'''DELETE GIVEN FEATURE VECTOR FROM FEATURE AND CONCETENATE WITH NEW DATA '''
def ManipulateFeatures(tX, data,features):
    tX = np.delete(tX, features, 1)
    return np.hstack((tX, data))

'''IMPUTE DATA WITH MEANS'''
def ImputeData(tX):
    tX = np.where(tX == -999, np.nan, tX)
    #Remove all columns with NAN
    tX = tX[:, ~np.all(np.isnan(tX), axis=0)]
    '''RM HIGHLY CORRELATED FEATURES'''
    tX = tX[:, ~np.all(tX[1:] == tX[:-1], axis=0)]
    '''MEAN EXCLUDING NAN VALUES'''
    tX_mean = np.nanmean(tX, axis=0)
    '''NAN = MEAN'''
    tX[np.where(np.isnan(tX))] = np.take(tX_mean, np.where(np.isnan(tX))[1])
    print(tX.shape)
    return tX

'''STANDARDIZE DATA'''
def Standardize(tX):
    '''MEAN OF DATA'''
    mean_x = np.mean(tX, axis=0)
    tX = tX - mean_x
    '''STANDARD DEVIATION'''
    std_x = np.std(tX, axis=0)
    tX[:, std_x > 0] = tX[:, std_x > 0] / std_x[std_x > 0]
    return tX, mean_x, std_x

'''PREPROCESS'''
def PreProcess(tX, rr_ = True):
    '''FEATURES PICKED BY HAND FOR LOG TRANSFORM'''
    '''MONOTONIC TRANSFORMATION'''
    log_feature_vec = np.array(([0, 2, 5, 9, 13, 16, 19, 21, 23, 26, 29]))
    tX = LogTransformData(tX, log_feature_vec)
    #print(tX.shape)
    '''CLEAN AND IMPUTE DATA'''
    tX = ImputeData(tX)
    if rr_:
        '''RIDGE REGRESSION AND LEAST SQUARES GIVE BETTER RESULTS IN THE CASE THAT WE ADD NON-LINEAR FEATURES AFTER STANDARDIZATION'''
        tX = Standardize(tX)[0]
        tX = AddFeatures(tX)
    else:
        '''IN CASE OF ITERATIVE ALGORITHMS, ADDING FEATURES AFTER STANDARDIZATION MAY RESULT IN FEATURES THAT DIVERGE DURING OPERATIONS SUCH AS SQUARE'''
        '''THEREFORE ADD FEATURES FIRST, AND STANDARDIZE EVERYTHING AT THE END'''
        tX = AddFeatures(tX)
        tX = Standardize(tX)[0]
    return tX

'''DATASET SEPERATED IN TERMS OF CATEGORIES IN COLUMN 23 (1-30)'''
def Categorize_Train(y, tX, ids):
    '''INDEICES FOR KEEPING SPECIFIC INDICES OF DATA POINTS'''
    ind = [[] for j in range(3)]
    xx = [[] for j in range(3)]
    yy = [[] for j in range(3)]
    iids = [[] for j in range(3)]
    
    '''COLUMN 23 CATEGORICAL DATA: SEPERATE DATA INTO CATEGORIES ACCORDINGLY'''
    '''CATEGORY 2 & 3 HAVE LOW AMOUNT OF TRAINING EXAMPLES'''
    '''CATEGORY 2 AND 3 CONCETENATED'''
    for i in range(3): 
        ind[i] = np.nonzero(tX[:, 22] == i)[0]
        if i == 2:
            ind[i] = np.hstack((ind[i], np.nonzero(tX[:, 22] == (i+1))[0].T))       
        xx[i] = tX[ind[i]]
        yy[i] = y[ind[i]]
        iids[i] = ids[ind[i]]
        
    return np.array((yy)), np.array((xx)), np.array((iids)), np.array((ind))

'''CATEGORIZE TEST'''
'''SAME AS IN TRAINING CATEGORIZATION BUT WE DONT HAVE Y ARRAY'''
def Categorize_Test(tX, ids):
    '''CATEGORY 2 AND 3 CONCETENATED'''
    ind = [[] for j in range(3)]
    xx = [[] for j in range(3)]
    iids = [[] for j in range(3)]
    
    for i in range(3): 
        ind[i] = np.nonzero(tX[:, 22] == i)[0]
        if i == 2:
            ind[i] = np.hstack((ind[i], np.nonzero(tX[:, 22] == (i+1))[0].T))   
        xx[i] = tX[ind[i]]
        iids[i] = ids[ind[i]]
        
    return np.array((xx)), np.array((iids)), np.array((ind))

'''PREDICTIONS INTO COMPARABLE FORM'''
'''PREDICTIONS OF EACH CATEGORY CONCETENATED ACCORDING TO THEIR ORIGINAL INDICES'''
def Decategorize(y_cat, ind):
    size = y_cat[0].shape[0] + y_cat[1].shape[0] + y_cat[2].shape[0]
    y = np.zeros((size,), dtype=np.float)
    for i in range(len(y_cat)):
        y[ind[i]] = y_cat[i]
    return y

'''CHECK VALIDATION SCORE'''
'''NORMALLY, SCORES FOR EACH CATEGORY ARE DISTINCT'''
'''THIS FUNCTION PROVIDES A TOTAL ACCURACY BY DIVIDING TOTAL (PRED == TARGET) TO TOTAL TARGET AMOUNT'''
def WeightedAverage(pred, target):
    total_count = pred[0].shape[0] + pred[1].shape[0] + pred[2].shape[0]
    true_count = 0
    for i in range(3):
        true_count +=  np.sum(pred[i] == target[i])
    acc = true_count / total_count
    return acc
'''FEATURE CORRELATION MAP: ONLY FOR VISUALIZATION'''
'''CORRELATED FEATURES: CORR > THRESHOLD : USE FOR SYNTHESIS'''
def CorrMap(tX):
    df = pd.DataFrame(tX)
    corr = df.corr()
    return corr.style.background_gradient(cmap='coolwarm')

'''RANDOM DATA SPLIT'''
'''SEED PROVIDES A FIXED RANDOMNESS, BUT IS NOT USED AS WE WANT FULL RANDONMNESS'''
def RandomizedDataSplit(tX, y, ids, inds, split_size = 0.1, my_seed=42):
    '''SET SEED FOR REMOVING RANDOMNESS'''
    np.random.seed(my_seed)
    '''RANDOM INDEXES'''
    size = y.shape[0]
    ind = np.random.permutation(size)
    split = int(np.floor(split_size * size))
    
    ind_train = ind[split:]
    ind_valid = ind[:split]
    
    
    '''SPLIT DATA ACCORDING TO RANDOM INDICES'''
    tX_train = tX[ind_train]
    tX_valid = tX[ind_valid]
    y_train = y[ind_train]
    y_valid = y[ind_valid]
    ids_train = ids[ind_train]
    ids_valid = ids[ind_valid]
    inds_train = inds[ind_train]
    inds_valid = inds[ind_valid]
    
    print("Shapes of tX, y, Ids & Indices for Training: ", tX_train.shape, y_train.shape, ids_train.shape, inds_train.shape)
    print("Shapes of tX, y, Ids & Indices for Validation: ", tX_valid.shape, y_valid.shape, ids_valid.shape, inds_valid.shape)
    return (tX_train, y_train, ids_train, inds_train),(tX_valid, y_valid, ids_valid, inds_valid)

'''BACKWARD SELECTION METHOD FOR BEST FEATURE SELECTION: GREEDY APPROACH'''
'''REMOVE LEAST EFFECTING FEATURES'''
def BackwardSelection(y, tX, tX_valid, y_valid, model = "RR"):
    selected_features = []
    cur_best_acc = 0
    improved = True     
    while improved:
        improved = False
        worst_ft = -1 
        for i in range(tX.shape[1]):
            if i not in selected_features:
                
                diff = set(list(range(tX.shape[1]))) - set(selected_features + [i])            
                #print(tX[:,list(diff)].shape,y.shape)
                
                cur_acc = CrossValidation(y, tX[:,list(diff)],10)
                #print(cur_acc)
                if cur_best_acc <= cur_acc:
                    #print("best so far: ",cur_best_acc)
                    improved = True
                    cur_best_acc = cur_acc
                    worst_ft = i                    
        if improved:
            selected_features.append(worst_ft)  
            
    return list(set(list(range(tX.shape[1]))) - set(selected_features )), cur_best_acc

'''FORWARD OF BACKWARD SELECTION'''
def Selection(y, tX, tX_valid, y_valid, typ = "BS", model = "RR"):
    if typ == "BS":
        selected_features, cur_best_acc = BackwardSelection(y, tX, tX_valid, y_valid,model)
        return selected_features, cur_best_acc
    elif typ == "FS":
        selected_features, cur_best_acc = ForwardSelection(y, tX, tX_valid, y_valid,model) 
        return selected_features, cur_best_acc
    
'''FORWARD SELECTION METHOD FOR BEST FEATURE SELECTION: GREEDY APPROACH'''
def ForwardSelection(y, tX, tX_valid, y_valid, model = "RR"):    
    selected_features = []
    cur_best_acc = 0  
    improved = True
    while improved:
        
        improved = False
        best_ft = -1 
        for i in range(tX.shape[1]):
            if i not in selected_features: 
                #calculate accuracy             
                cur_acc = CrossValidation(y, tX[:,selected_features+[i]],5)
                #print(cur_acc)
                #accuracy is improved
                if cur_best_acc <= cur_acc:
                    improved = True                   
                    cur_best_acc = cur_acc
                    best_ft = i                 
                    
        if improved:
            selected_features.append(best_ft)
            #print(selected_features)
         
    return selected_features, cur_best_acc

'''ADD NEW FUTURES'''
'''NON-LINEAR FUNCTIONS ADDED SUCH AS COS, SIN, SQUARE, ETC.'''
def AddFeatures(tX):
    prime_numbers = [2,3]
    #ADD COS / SIN , SQRT 
    #CHECK FEATURE SYNTHESIS
    pm = 0
    '''ADD COSINUS AND SINUS OF FEATURES'''
    loop_count = tX.shape[1]
    for i in range(loop_count):
            tX = np.hstack((tX, np.cos(tX[:,i]).reshape(-1,1)))
    for i in range(loop_count):
            tX = np.hstack((tX, np.sin(tX[:,i]).reshape(-1,1)))
    
    '''ADD SQUARE AND POWER TO THE THREE'''
    for pm in range(len(prime_numbers)):
        for i in range(3*loop_count):
            tX = np.hstack((tX, np.power(tX[:,i], prime_numbers[pm]).reshape(-1,1)))
    
    return tX

'''CROSS VALIDATION HELPER FUNCTION'''
'''SEPERATE DATA ACCORDING TO K-FOLD SIZE INTO K-FOLD TIMES DIFFERENT TRAINING VALIDATION PARTITIONS'''
def SelectIndices(y, k_fold, seed):
    row_count = y.shape[0]
    window_size = int((row_count / k_fold))
    remainder = row_count % k_fold
    '''SEED IN TERMS OF SHUFFLING ONLY ONCE'''
    np.random.seed(seed)
    rand_indices = np.random.permutation(row_count)
    indices = [[] for i in range(k_fold)]
    
    '''WINDOW SIZE IS USED FOR DATA PARTITION'''
    '''EVERY INDEX CORRESPOND TO CHUNK OF DATA: ONE WILL BE CHOSEN AS VALIDATION AND REST FOR TRAINING'''
    '''DIFFERENT VALIDATION AT EVERY SPLIT'''
    for k in range(k_fold):        
            indices[k] = np.array((rand_indices[k*window_size:(k+1)*window_size]))
            
    return np.array(indices)

'''CROSS VALIDATION'''
'''INSTEAD OF A SPECIFIC TRAIN/VALID SPLIT, K-FOLD AMOUNT OF TRAIN/VALID SPLIT CREATED'''
def CrossValidation(y, tX, k, cat_, lambda_,seed = 42):
    #seed = np.random.randint(10)
    indices = SelectIndices(y,k,seed)
    average_acc = 0
    w_vec = list()
    '''K-FOLD SPLITS ARE CREATED'''
    for i in range (k): 
        tr_indices = list()
        count = 1
        for tr_ in range(len(indices)):
            if i != tr_:
                if count == 1:
                    tr_indices = np.array((indices[tr_]))
                    count += 1
                else:
                    tr_indices = np.hstack((tr_indices, indices[tr_]))
                
        #print(tr_indices)
        valid_indices = indices[i]   
        xk_train = tX[tr_indices]
        xk_valid = tX[valid_indices]
        yk_train = y[tr_indices]
        yk_valid = y[valid_indices]
        #print(yk_train.shape,xk_train.shape)
        #print(xk_valid.shape)

        '''SPLITS ARE USED FOR TRAINING AND VALIDATION'''
        init_w_gd = np.array((InitWeights(xk_train.shape[1])))
        (w,loss) = ridge_regression(yk_train, xk_train, lambda_)
        ls_tr_pred = predict_labels(w, xk_valid)
        average_acc += (ls_tr_pred == yk_valid).mean()/k
        '''EACH TRAINING CREATES A WEIGHT MATRIX, EACH SAVED'''
        w_vec.append(w)
    '''MEAN OF WEIGHT MATRICES ARE TAKEN FOR A MORE REILABLE WEIGHT MATRIX'''
    '''CORRESPONDING ACCURACY IS MORE RELIABLE THAN A CERTAIN RANDOM DATA SPLIT'''
    print("Cross Validation Accuracy for Category ",cat_,": ",average_acc)
    w_final = w_vec[0]
    for weight in range(1,len(w_vec)):
        w_final += w_vec[weight]
    w_final = w_final / len(w_vec)
    
    return w_final, average_acc
    
'''BUILD FULL DATA MODEL WITH CATEGORIZATION'''
def BuildDataModel_Train(y, tX_old, ids, pp = False):
    '''CATEGORIZE DATA'''
    y_cat, tX_cat, id_cat, ind_cat = Categorize_Train(y, tX_old, ids)  
    '''PREPROCESS EACH CATEGORY'''
    '''SET UP FOR ITERATIVE METHODS: IN CASE OF LEAST SQUARES WITH NORMAL EQUATIONS OR RIDGE REGRESSION CHANGE ALST PARAMETER TO TRUE'''
    for cat_ in range(tX_cat.shape[0]):
        tX_cat[cat_] = PreProcess(tX_cat[cat_], pp)  
         
    '''TRAIN SET'''
    tX_tr_cat = [[] for j in range(3)]
    y_tr_cat = [[] for j in range(3)]
    id_tr_cat = [[] for j in range(3)]
    ind_tr_cat = [[] for j in range(3)]

    '''VALID SET'''
    tX_val_cat = [[] for j in range(3)]
    y_val_cat = [[] for j in range(3)]
    id_val_cat = [[] for j in range(3)]
    ind_val_cat = [[] for j in range(3)]

    '''RANDOMLY SPLIT DATA TO TRAIN/VALIDATION: DEAULT RATIO 0.1'''
    for i in range(len(tX_cat)):
        (tX_tr_cat[i], y_tr_cat[i],id_tr_cat[i],ind_tr_cat[i]), (tX_val_cat[i], y_val_cat[i],id_val_cat[i], ind_val_cat[i]) = RandomizedDataSplit(tX_cat[i], y_cat[i], id_cat[i], ind_cat[i])

    '''CONVERT TRAIN AND DATASET INTO NUMPY ARRAYS'''
    tX_tr_cat = np.array((tX_tr_cat))
    y_tr_cat = np.array((y_tr_cat))
    id_tr_cat = np.array((id_tr_cat))
    ind_tr_cat = np.array((ind_tr_cat))

    tX_val_cat = np.array((tX_val_cat))
    y_val_cat = np.array((y_val_cat))
    id_val_cat = np.array((id_val_cat))
    ind_val_cat = np.array((ind_val_cat))

    return (y_tr_cat, tX_tr_cat, id_tr_cat, ind_tr_cat), (y_val_cat, tX_val_cat, id_val_cat, ind_val_cat)

'''CROSS VALIDATION DATA MODEL'''
'''NO SPLIT REQUIRED AS CROSSVALIDATION WILL BE USED LATER ON FOR TRAIN/VALID SPLIT'''
def BuildDataModel_CV(y, tX_old, ids):
    '''CATEGORIZE DATA'''
    y_cat, tX_cat, id_cat, ind_cat = Categorize_Train(y, tX_old, ids)
    
    '''PREPROCESS EACH CATEGORY'''
    '''SET UP FOR ITERATIVE METHODS: IN CASE OF LEAST SQUARES WITH NORMAL EQUATIONS OR RIDGE REGRESSION CHANGE ALST PARAMETER TO TRUE'''
    for cat_ in range(tX_cat.shape[0]): 
        tX_cat[cat_] = PreProcess(tX_cat[cat_], True)
        
    return y_cat, tX_cat, id_cat, ind_cat

'''BUILD FULL DATA MODEL WITH CATEGORIZATION'''
'''NO TARGET ARRAY PROVIDED FOR ACTUAL TESTING DATA'''
def BuildDataModel_Test(tX_old, ids , pp = False):
    '''CATEGORIZE DATA'''
    tX_cat, id_cat, ind_cat = Categorize_Test(tX_old, ids)
    
    for cat_ in range(tX_cat.shape[0]):
        tX_cat[cat_] = PreProcess(tX_cat[cat_], False)
        
    '''CONVERT TRAIN AND DATASET INTO NUMPY ARRAYS'''
    tX_cat = np.array((tX_cat))
    id_cat = np.array((id_cat))
    ind_cat = np.array((ind_cat))
    
    return tX_cat, id_cat, ind_cat

'''BUILD FULL DATA MODEL WITH CATEGORIZATION -> FOR RIDGE REGRESSION AND LEAST SQUARES'''
'''NO TARGET ARRAY PROVIDED FOR ACTUAL TESTING DATA'''
def BuildDataModel_CV_Test(tX_old, ids):
    tX_cat, id_cat, ind_cat = Categorize_Test(tX_old, ids)
    
    for cat_ in range(tX_cat.shape[0]):
        tX_cat[cat_] = PreProcess(tX_cat[cat_])
        
    '''CONVERT TRAIN AND DATASET INTO NUMPY ARRAYS'''
    tX_cat = np.array((tX_cat))
    id_cat = np.array((id_cat))
    ind_cat = np.array((ind_cat))
    
    return tX_cat, id_cat, ind_cat

######################## DATA ENGINEERING ################################
##########################################################################


######################## COST && GRADIENT ################################
##########################################################################

'''MEAN SQUARED ERROR OR MEAN ABSOLUTE ERROR'''
'''LOSS CALCULATION FOR BATCH GD OR STOCHASTIC GD'''
def compute_loss(y, tx, w, typ):
    '''typ = <LOSS_TYPE(WITH CAPITAL LETTERS)>'''
    loss = 0
    N = y.shape[0]
    if typ == "MSE":
        loss = (1/(2*N))*np.sum(np.square(y - (tx@w)))
    elif typ == "MAE":
        loss = (1/(2*N))*np.sum(np.abs(y - (tx@w)))
    return loss

'''GRADIENT CALCULATION FOR BATCH GD'''
def compute_gradient(y, tx, w):
    N = y.shape[0]
    e = y - tx@w
    grad = (-1/N) * (tx.T@e)
    return grad

'''GRADIENT CALCULATION FOR STOCHASTIC GD'''
def compute_stoch_gradient(y, tx, w):
    N = y.shape[0]
    e = y - tx@w
    grad = (-1/N)*(tx.T@e)
    return grad

'''LEAST SQUARES WITH NORMAL EQUATIONS LOSS COMPUTATION'''
def compute_ls_loss(y, tx, w):
    loss = 0
    N = y.shape[0]
    loss = (1/(2*N))*(tx.T@(y - tx@w))

'''RIDGE REGRESSION(REGULARIZED LEAST SQUARES WITH NORMAL EQUATION) LOSS COMPUTATION'''    
def compute_rdg_loss(y, tx, w, lambda_):
    loss = 0
    N = y.shape[0]
    loss = (1/(2*N))*np.sum(np.square(y - (tx@w))) + (lambda_*np.sum(w.T@w))
    return loss

'''SIGMOID CALCULATION'''
def sigmoid(tx, w):
    z = 1 / (1 + np.exp(-1*(tx@w)))
    return z

'''LOGISTIC REGRESSION LOSS FUNCTION'''
def compute_log_loss(y, tx, w):
    loss = 0;
    sigm = sigmoid(tx,w)
    sigm2 = 1 - sigm
    N = y.shape[0]
    '''GIVEN THAT WE HAVE A VALUE THAT IS NEGATIVE OR REALLY SMALL(PYTHON CONVERTS IT TO ZERO DURING COMPUTATION)'''
    '''CONVERT THEM TO 1e-50'''
    sigm[sigm < 1e-50] = 1e-50
    sigm2[sigm2 < 1e-50] = 1e-50
    
    loss = (-1/N)*np.sum(y.T@np.log(sigm) + ((1-y).T@np.log(sigm2)))
    
    return loss

'''GRADIENT COMPUTATION FOR LOGISTIC REGRESSION'''
def compute_log_gradient(y, tx, w):
    N = y.shape[0]
    z = sigmoid(tx,w)
    grad = (1/N) * (tx.T@(z - y))
    return grad

'''LOGISTIC REGRESSION LOSS WITH REGULARIZATION'''
def compute_reg_log_loss(y, tx, w, lambda_):
    loss = 0;
    sigm = sigmoid(tx,w)
    sigm2 = 1 - sigm
    N = y.shape[0]
    '''GIVEN THAT WE HAVE A VALUE THAT IS NEGATIVE OR REALLY SMALL(PYTHON CONVERTS IT TO ZERO DURING COMPUTATION)'''
    '''CONVERT THEM TO 1e-50'''
    sigm[sigm < 1e-50] = 1e-50
    sigm2[sigm2 < 1e-50] = 1e-50
    loss = (-1/N)*(np.sum(y.T@np.log(sigm) + ((1-y).T@np.log(sigm2))) + ((lambda_/2)*np.sum(w.T@w)))
    
    return loss

'''LOGISTIC REGRESSION GRADIENT COMPUTATION WITH REGULARIZATION'''
def compute_reg_log_gradient(y, tx, w, lambda_):
    N = y.shape[0]
    z = sigmoid(tx,w)
    grad = (1/N) * ((tx.T@(z - y)) + (lambda_*w))
    return grad

######################## COST && GRADIENT ################################
##########################################################################

############################# MODELS #####################################
##########################################################################

'''BATCH GRADIENT DESCENT'''
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w, "MSE")
        grad = compute_gradient(y, tx, w)
        w = w - (gamma * grad)
        if (n_iter % 100) == 0:
            print("Gradient Descent({bi}/{ti}) loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return (w, loss)

'''STOCHASTIC GRADIENT DESCENT'''
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w 
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            loss = compute_loss(minibatch_y, minibatch_tx, w, "MSE")
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
            if (n_iter % 100) == 0:
                print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))          
    return (w, loss)

'''COMPUTE W_STAR: WEIGHT FOR NORMAL EQUATIONS BY LINEAR EQUATION SOLVER'''
def least_squares(y, tx):
    w_star = np.linalg.solve(tx.T@tx, tx.T@y)
    loss = compute_ls_loss(y, tx, w_star)
    return (w_star,loss)

'''RIDGE REGRESSION WITH LAMBDA PARAMETER AS REGULARIZATION PARAMETER'''
def ridge_regression(y, tx, lambda_):
    N = y.shape[0]
    a = tx.shape[0]
    m = (tx.T@tx)+(lambda_/(2*N))*np.identity(tx.shape[1])
    i = np.eye(m.shape[0],m.shape[0])
    w_ridge = np.linalg.lstsq(m,i)[0]@tx.T@y
    loss = compute_rdg_loss(y, tx, w_ridge, lambda_)
    return (w_ridge, loss)

'''LOGISTIC REGRESSION WITH BATCH GRADIENT DESCENT OR STOCHASTIC GRADIENT DESCENT'''
def logistic_regression(y, tx, initial_w, max_iters, gamma, mod = 1):
    if mod == 1:
        '''FOR GRADIENT DESCENT'''
        w = initial_w
        for n_iter in range(max_iters):
            loss = compute_log_loss(y, tx, w)
            grad = compute_log_gradient(y, tx, w)
            w = w - (gamma * grad)
            if (n_iter % 100) == 0:
                print("Logistic Regression Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

        return (w, loss)
    else:
        '''FOR STOCHASTIC GRADIENT DESCENT'''
        w = initial_w 
        for n_iter in range(max_iters):
            for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
                loss = compute_log_loss(minibatch_y, minibatch_tx, w)
                grad = compute_log_gradient(minibatch_y, minibatch_tx, w)
                w = w - gamma * grad
                print("Logistic Regression Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
        return (w, loss)

'''REGULARIZED LOGISTIC REGRESSION WITH BATCH GRADIENT DESCENT OR STOCHASTIC GRADIENT DESCENT'''
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, mod = 1):
    if mod == 1:
        '''FOR GRADIENT DESCENT'''
        w = initial_w
        for n_iter in range(max_iters):
            loss = compute_reg_log_loss(y, tx, w, lambda_)
            grad = compute_reg_log_gradient(y, tx, w, lambda_)
            w = w - (gamma * grad)
            if (n_iter % 100) == 0:
                print("Regularized Logistic Regression Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

        return (w, loss)
    else:
        '''FOR STOCHASTIC GRADIENT DESCENT'''
        w = initial_w 
        for n_iter in range(max_iters):
            for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
                loss = compute_reg_log_loss(minibatch_y, minibatch_tx, w, lambda_)
                grad = compute__reg_log_gradient(minibatch_y, minibatch_tx, w, lambda_)
                w = w - gamma * grad
                print("Regularized Logistic Regression Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
        return (w, loss)
############################# MODELS #####################################
##########################################################################


########################### RUN MODELS ###################################
##########################################################################

'''PREDICTIONS FOR MODELS WITHOUT CROSS VALIDATION'''
'''CHOOSE MOD FOR LEARNING ALGORITHM: 1: LS-GD, 2: LS-SGD, 3: NORMAL EQ, 4: RR, 5: LR, 6: RLR'''
def Main(y_cat, tX_cat, y_val_cat, tX_val_cat, y, tX_old, ids, mod = 4):
    init_w_gd = np.array((InitWeights(tX_cat[0].shape[1]),InitWeights(tX_cat[1].shape[1]),InitWeights(tX_cat[2].shape[1])))
    gd_tr_pred = np.copy((y_val_cat))
    w_gd = np.copy((init_w_gd))
    max_iters, epochs, gamma, lambda_,decay,k_fold = HyperParameters()
    cat_lst = [0, 1, 2]
    
    if mod == 1:
        for epoch_ in range(epochs):

            start = timeit.default_timer()

            for cat_ in cat_lst:   
                (w_gd[cat_],loss1) = least_squares_GD(y_cat[cat_], tX_cat[cat_],w_gd[cat_], max_iters, gamma)

            stop = timeit.default_timer()

            print('Time: ', stop - start) 

            gamma = GammaScheduler(gamma, decay, epoch_)
    
    elif mod == 2: 
        
         for epoch_ in range(epochs):

            start = timeit.default_timer()
            gamma = 1e-5
            for cat_ in cat_lst:   
                (w_gd[cat_],loss1) = least_squares_SGD(y_cat[cat_], tX_cat[cat_],w_gd[cat_], max_iters, gamma)

            stop = timeit.default_timer()

            print('Time: ', stop - start) 

            gamma = GammaScheduler(gamma, decay, epoch_)
            
    elif mod == 3:
        print("Modifying Data in terms of Least Square with Normal Equations: ")
        (y_cat, tX_cat, ids_cat, ind_cat), (y_val_cat, tX_val_cat, ids_val_cat, ind_val_cat) = BuildDataModel_Train(y,tX_old,ids, True)
        for cat_ in cat_lst:   
            (w_gd[cat_],loss1) = least_squares(y_cat[cat_], tX_cat[cat_])

    elif mod == 4: 
        print("Modifying Data in terms of Ridge Regression: ")
        (y_cat, tX_cat, ids_cat, ind_cat), (y_val_cat, tX_val_cat, ids_val_cat, ind_val_cat) = BuildDataModel_Train(y,tX_old,ids,True)
        for cat_ in cat_lst:   
            (w_gd[cat_],loss1) = ridge_regression(y_cat[cat_], tX_cat[cat_],lambda_)
            
    elif mod == 5:
        
         for epoch_ in range(epochs):

            start = timeit.default_timer()

            for cat_ in cat_lst:   
                (w_gd[cat_],loss1) = logistic_regression(y_cat[cat_], tX_cat[cat_],w_gd[cat_], max_iters, gamma)

            stop = timeit.default_timer()

            print('Time: ', stop - start) 

            gamma = GammaScheduler(gamma, decay, epoch_)
            
    elif mod == 6:
        
         for epoch_ in range(epochs):

            start = timeit.default_timer()

            for cat_ in cat_lst:   
                (w_gd[cat_],loss1) = reg_logistic_regression(y_cat[cat_], tX_cat[cat_], lambda_, w_gd[cat_], max_iters, gamma)

            stop = timeit.default_timer()

            print('Time: ', stop - start) 

            gamma = GammaScheduler(gamma, decay, epoch_)
            
    '''PREDICTIONS'''
    for cat_ in cat_lst:
        gd_tr_pred[cat_] = predict_labels(w_gd[cat_], tX_val_cat[cat_])
        print(gd_tr_pred[cat_])
            
    acc = WeightedAverage(gd_tr_pred, y_val_cat)
    print("Accuracy of Model:", acc)
    return w_gd


'''CROSS VALIDATION WEIGHT'''
'''WEIGHTS ARE TAKEN AS MEAN OF VARIOUS MEANS BY TRAINING ON DIFFERENT TRAIN-VALID PARTITIONS'''
'''ACCURACY INCREASE IS NOT THE PURPOSE BUT WEIGHT AND ACCURACY VALIDATION ARE FIXED'''
def CV_Main(y_cat, tX_cat):
    cat_lst = [0, 1, 2]
    max_iters, epochs, gamma, lambda_, decay, k_fold = HyperParameters()
    w_res = np.array((InitWeights(tX_cat[0].shape[1]),InitWeights(tX_cat[1].shape[1]),InitWeights(tX_cat[2].shape[1])))
    cv_tr_pred = [[] for i in range(len(cat_lst))]
    for cat_ in cat_lst:
        w_final, avg_acc = CrossValidation(y_cat[cat_], tX_cat[cat_], k_fold, cat_, lambda_)
        w_res[cat_] = w_final
        print("Final weight vector shape: ",w_final.shape)
    return w_res

'''FUNCTION FOR TESTING'''
'''DECATEGORIZES PREDICTIONS AFTER PREDICTIONS'''
def Tester(w_cv, tX_test_cat, id_test_cat, ind_test_cat, mod = 1):
    OUTPUT_PATH = 'results'+ str(mod)+'.csv'
    cat_lst = [0, 1, 2]
    y_pred = [[] for i in range(3)]
    for cat_ in cat_lst:
        y_pred[cat_] = predict_labels(w_cv[cat_], tX_test_cat[cat_])
    y_pred = np.array((y_pred))
    pred_vec = Decategorize(y_pred, ind_test_cat)
    pred_ids = Decategorize(id_test_cat, ind_test_cat)
    create_csv_submission(pred_ids, pred_vec, OUTPUT_PATH)
    
########################### RUN MODELS ###################################
##########################################################################


########################### FEATURE 30 ###################################
############################# TRIAL ######################################

'''CATEGORIZE TRAIN INTO TWO CATEGORIES ACCORDING TO FEATURE 30 - TARGET RELATION'''
def Categorize_Lucky(y, tX, ids):
    ind = [[] for j in range(2)]
    xx = [[] for j in range(2)]
    yy = [[] for j in range(2)]
    iids = [[] for j in range(2)]
    
    ind[0] = np.nonzero(tX[:, 29] <= 0.1)[0]      
    xx[0] = tX[ind[0]]
    yy[0] = y[ind[0]]
    iids[0] = ids[ind[0]]
    
    ind[1] = np.nonzero(tX[:, 29] > 0.1)[0]      
    xx[1] = tX[ind[1]]
    yy[1] = y[ind[1]]
    iids[1] = ids[ind[1]]
        
    return np.array((yy)), np.array((xx)), np.array((iids)), np.array((ind))



'''CATEGORIZE TEST INTO TWO CATEGORIES ACCORDING TO FEATURE 30 - TARGET RELATION'''
def Categorize_Lucky_Test(tX, ids):
    '''CATEGORY 2 AND 3 CONCETENATED'''
    ind = [[] for j in range(2)]
    xx = [[] for j in range(2)]
    iids = [[] for j in range(2)]
    
    ind[0] = np.nonzero(tX[:, 29] <= 0.1)[0]      
    xx[0] = tX[ind[0]]
    iids[0] = ids[ind[0]]
    
    ind[1] = np.nonzero(tX[:, 29] > 0.1)[0]      
    xx[1] = tX[ind[1]]
    iids[1] = ids[ind[1]]
        
    return np.array((xx)), np.array((iids)), np.array((ind))

'''BUILD TRAIN DATA MODEL FOR THE FEATURE 30'''
'''CROSS VALIDATION DATA MODEL FOR FEATURE 30'''
def BuildDataModel_Lucky(y, tX_old, ids):
    y_cat, tX_cat, id_cat, ind_cat = Categorize_Lucky(y, tX_old, ids)
    
    for cat_ in range(tX_cat.shape[0]): 
        tX_cat[cat_] = PreProcess(tX_cat[cat_], True)
        
    return y_cat, tX_cat, id_cat, ind_cat

'''BUILD TEST DATA MODEL FOR THE FEATURE 30'''
def BuildDataModel_Lucky_Test(tX_old, ids):
    tX_cat, id_cat, ind_cat = Categorize_Lucky_Test(tX_old, ids)
    
    for cat_ in range(tX_cat.shape[0]):
        tX_cat[cat_] = PreProcess(tX_cat[cat_])
        
    '''CONVERT TRAIN AND DATASET INTO NUMPY ARRAYS'''
    tX_cat = np.array((tX_cat))
    id_cat = np.array((id_cat))
    ind_cat = np.array((ind_cat))
    
    return tX_cat, id_cat, ind_cat

'''PREDICTIONS INTO COMPARABLE FORM'''
def Decategorize_Lucky(y_cat, ind):
    size = y_cat[0].shape[0] + y_cat[1].shape[0]
    y = np.zeros((size,), dtype=np.float)
    for i in range(len(y_cat)):
        y[ind[i]] = y_cat[i]
    return y

'''TEST FEATURE 30 MODEL'''
def Lucky_Main(y_cat, tX_cat):
    cat_lst = [0, 1]
    max_iters, epochs, gamma, lambda_, decay, k_fold = HyperParameters()
    w_res = np.array((InitWeights(tX_cat[0].shape[1]),InitWeights(tX_cat[1].shape[1])))
    cv_tr_pred = [[] for i in range(len(cat_lst))]
    for cat_ in cat_lst:
        w_final, avg_acc = CrossValidation(y_cat[cat_], tX_cat[cat_], k_fold, cat_, lambda_)
        w_res[cat_] = w_final
        print("Final weight vector shape: ",w_final.shape)
    return w_res

############################# TRIAL ######################################
########################### FEATURE 30 ###################################



