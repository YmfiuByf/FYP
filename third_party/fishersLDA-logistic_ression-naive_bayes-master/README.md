# fishersLDA-logistic_ression-naive_bayes
Implementation of Fishers Linear Discrimination Analysis, Logistic Regression and Naive Bayes with Marginal Gaussian Distributions

Readme file
Author Name - Rahul Biswas

Softwares required - 
python3
pandas 
numpy 
scipy
matplotlib

The project contains the following python files to compile - 

1. LDA1dProjection.py - This file takes the Boston Housing data and finds the 2 class FLDA and then plots the points. 
2. LDA2dGaussGM.py - This file takes the Digits data, and uses Gaussian approximation to classify after projecting the data to 2 dimensions using Fisher’s LDA
3. logisticregression.py - This file implements the Logistic Regression for both the datasets using the IRLS using WiKipedia as a reference. The constants are hard-coded and not tuned using a script. (Constant decided after compiling for around 20 different constants)
4. naiveBayesGaussian - This files implements the Naive Bayes for both the datasets.


Compiling instructions - 

Usage: python3 LDA1dProjection.py /path/to/dataset.csv num_crossval
Example - python3 LDA1dProjection.py Boston.csv 10

Usage:  python3 LDA2dGaussGM.py /path/to/dataset.csv num_crossval
Example - python3 LDA2dGaussGM.py digits.csv 10


Usage: python3 logisticregression.py /path/to/dataset.csv num_splits train_percent
Example - python3 logisticregression.py boston.csv 10 "10 25 50 75 100"


Usage: python3 naiveBayesGaussian.py /path/to/dataset.csv num_splits train_percent
Example - python3 naiveBayesGaussian.py digits.csv 10 "10 25 50 75 100"

Note - The plt.show() is working perfectly at my personal computer, but didn’t work when I tried with the cse-lab computer. I think there is something wrong with the matplotlib.back_end. Please do let me know if this is an issue and I can show you how it works on my computer. 
