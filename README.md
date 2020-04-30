MEMORY LEAK: HIGGS BOSON CHALLENGE:
*Baris Sevilmis: baris.sevilmis@epfl.ch

*Berk Mandiracioglu: berk.mandiracioglu@epfl.ch

*Onur Veyisoglu: onur.veyisoglu@epfl.ch
Submission ID: 23430 - Ridge Regression

IMPLEMENTATION.PY

1) In implementations.py, functions are grouped in subsections. DATA ENGINEERING, LOSS AND GRADIENTS, MODELS, RUN FUNCTIONS OF R TRAIN AND TEST AND FINALLY FEATURE 30 CATEGORIZATION. Last group was only created to test categorization on another feature(30th feature), therefore is not included in run.py.

2) DATA ENGINEERING: Functions are commented, and explained in a detailed manner. Variable names are provided as reasonable names. Modularity is highly used, as usage of functions reduces codes complexity and increases readability. BuildDataModel functions are the highest level data model builder functions. They are seperated for training and testing as well as CrossValidation Method and Randomized Split Methods:

-Training and Testing: Test Set has no target vector, therefore seperated.
-CV and Randomized Split: We have used Ridge Regression in Cross Validation as it provided best set of results for us. Cross Validation method in terms of weight vector determination is used to produce a more stabil and reliable weight vector as mean of k-fold training weight vectors are taken. Randomized split is used as a regular validation method.

Other than validation split, data is categorized and preprocessed before used for training. Ridge Regression gives best result if we add features after standardization, as iterative methods are providing best result vice versa. Therefore, preprocess calls standardization and add features methods vice versa. Rest of the methods are preprocess or categorization specific.

3) LOSS & GRADIENT: Loss computation and gradient computations for different models are included in this section. Names are provided clearly such that including comments all the functions should be clear in terms of their model.

4) MODELS: Implementations for each requested model are provided in this section.

5) RUN: Main, CV_Main and Tester functions are provided. Main function contains all the learning algorithms: given the last parameter in range of [1,6], specific learning algorithms will be called. For Ridge Regression and Least Squares, data are modified within function again.

RUN.PY

Requested import are made, Pandas is only for visualizaiton in terms of correlation map. It is still not used.

Data paths will be taken as inputs, if not need to modify commented train and test paths

Comment/Uncomment algorithm training lines if needed: 1 - Batch GD, 2 - SGD, 3 - Leas Squares, 4- Ridge Regression, 5 - Logistic Regression, 6 - Regularized Logistic Regression, 7 - Cross Validation Ridge Regression

Same is also applied for Tester lines where predictions are produced.

Resulting csv files are written in result+str(learning algorithm number)+.csv in the same location with the code.

All learning algorithms are being run with Tester including CV.

NOTE: PLEASE MODIFY DATA PATH, IF DON'T WANT TO ENTER AS INPUT: ENTER FULL PATH(use pwd) IN BOTH CASES
NOTE-2: ONLY THE BEST LEARNING ALGORITHM IS UNCOMMENTED, PLEASE REMOVE COMMENTS IF NEEDED
NOTE-3: DATA PROCESSING FOR 1-6 LEARNING ALGORITHMS ARE COMMENTED, PLEASE UNCOMMENT(ALSO TESTER & LEARNING LINES) IF WANT TO SEE RESULTS FOR EACH OF THEM 
