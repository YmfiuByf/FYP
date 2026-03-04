import naiveBayesGaussian
import logisticREGRESSION_new
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from random import seed
train_percent = [10, 25, 50, 75, 100]
# seed(5525)
# np.random.seed(5525)
e1,s1 = naiveBayesGaussian.nb('digits.csv', 10)
e2, s2 = logisticREGRESSION_new.logisticRegression('digits.csv', 10)
#

plt.xlim([9.5, 100.5])
# plt.ylim([0.05, 0.30])
plt.errorbar(train_percent, e2, s2, label='logistic regression', fmt='o-', capthick=2)
plt.errorbar(train_percent, e1, s1, label='naive bayes', fmt='o--', color='r', capthick=2)
plt.legend()
plt.ylabel('Test Error Rate')
plt.xlabel('Training Percent')
plt.title('Logistic Regression V.S. Naive Bayes for Digits', fontsize=12)
plt.show()