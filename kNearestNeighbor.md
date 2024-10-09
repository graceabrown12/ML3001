Q0

## 1. What is the difference between regression and classification? Regression means that features and covariates are used to predict a numeric. Classification is when you use features and covariates to predict a categorical variable. 
## 2. What is a confusion table? What does it help us understand about a model's performance? A confusion table will cross-tabulate the predicted or actual values. This helps us understand where the model is predicting things correctly or where it is making mistakes.
## 3. Define accuracy. Can an accurate model be flawed for practical use? Explain. Accuracy means that some labels are predicted correctly. A model can make systemic errors that outweigh its accuracy.
## 4. What does the SSE quantify about a particular model? SSE quantifies the squared error made by the model. 
## 5. What are overfitting and underfitting? Overfitting means that a model is too complex for the given data. Undercutting is when a model is too simple for the given data and unable to show important features. 
## 6. Why does splitting the data into training and testing sets, and choosing by evaluating accuracy or SSE on the test set, improve model performance? Splitting the data into training and testing sets means that the model can use data it hasn’t seen yet 
## 7. With classification, we can report a class label as a prediction or a probability distribution over class labels. Please explain the strengths and weaknesses of each approach. A strength is that class labels can be easily interpreted. A weakness is that it is not clear how certain predictions may be. 


Q5

# 1. Load the `./data/heart_failure_clinical_records_dataset.csv`. Are there any `NA`'s to handle? use `.drop()` to remove `time` from the dataframe.

import pandas as pd
import numpy as np
df = pd.read_csv('./data/heart_failure_clinical_records_dataset.csv')
print(df.shape)
print(df.describe())

#> No missing values, since 299 values for every variable and 299 observations in total.

df = df.drop('time',axis=1)

# 2. Make a correlation matrix. What variables are strongly associated with a death event?
print(df.corr())
print('The variables with the strongest correlation with `DEATH_EVENT` are age (.254), ejection_fraction (-.269), and serum_creatine (.294). ')

# 3. For the dummy variables `anaemia`, `diabetes`, `high_blood_pressure`, `sex`, and `smoking`, compute a summary table of `DEATH_EVENT` grouped by the variable. For which variables does a higher proportion of the population die when the variable takes the value 1 rather than 0?
vars = ['anaemia','diabetes','high_blood_pressure','sex','smoking']
for var in vars:
    print(df.loc[:,[var,'DEATH_EVENT']].groupby(var).describe())
    
#> > Let's look at the means for high_blood_pressure. For the proportion of the population that has HBP, the DEATH_EVENT average is .371, while for the proportion of the population that does not have HBP, the DEATH_EVENT average is only .294. That's a 27% increase in the frequency of death events. So HPB seems highly predictive. On the other hand, for sex, the mean values are almost the same for men and women, at .32, so the sex variable isn't a very powerful predictor of death events. Anaemia and high blood pressure seem like the strongest predictors.



# 4. On the basis of your answers from 2 and 3, build a matrix $X$ of the variables you think are most predictive of a death, and a variable $y$ equal to `DEATH_EVENT`.

y = df['DEATH_EVENT']
vars = ['age','ejection_fraction','serum_creatinine','high_blood_pressure','anaemia']
X = df.loc[:,vars]

# 5. Maxmin normalize all of the variables in `X`.

def maxmin(x):
    u = (x-min(x))/(max(x)-min(x))
    return u
X = X.apply(maxmin)

# 6. Split the sample into ~80% for training and ~20% for evaluation. (Try to use the same train/test split for the whole question, so that you're comparing apples to apples in the questions below.).

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=100)
np.random.seed(100) 
N = X.shape[0]
all = np.arange(1,N)
train = np.random.choice(N,int(.8*N) ) # Generate random indices for training set
test = [item for item in all if item not in train] # Find test indices

#test = np.where( train not in np.linspace(1,N) )
X_train = X.iloc[train,:]
y_train = y.iloc[train]
X_test = X.iloc[test,:]
y_test = y.iloc[test]

# 7. Determine the optimal number of neighbors for a $k$NN regression for the variables you selected.

from sklearn.neighbors import KNeighborsRegressor

# Determine the optimal k:
k_bar = 25
k_grid = np.arange(1,k_bar) # The range of k's to consider
SSE = np.zeros(k_bar) 

for k in range(k_bar):
    knn = KNeighborsRegressor(n_neighbors=k+1)
    predictor = knn.fit(X_train,y_train) 
    y_hat = knn.predict(X_test)
    SSE[k] = np.sum( (y_test-y_hat)**2 ) # Bug in sklearn requires .values

SSE_min = np.min(SSE) # highest recorded accuracy
min_index = np.where(SSE==SSE_min) 
k_star = k_grid[min_index] # Find the optimal value of k
print(k_star)

plt.plot(np.arange(0,k_bar),SSE) # Plot accuracy by k
plt.xlabel("k")
plt.title("optimal k:"+str(k_star)+', SSE:'+str(SSE_min))
plt.ylabel('SSE')
plt.show()

# 8. OK, do steps 5 through 7 again, but use all of the variables (except `time`). Which model has a lower Sum of Squared Error? Which would you prefer to use in practice, if you had to predict `DEATH_EVENT`s? If you play with the selection of variables, how much does the SSE change for your fitted model on the test data? Are more variables always better? Explain your findings.

X = df.drop('DEATH_EVENT',axis=1)

X_train = X.iloc[train,:]
y_train = y.iloc[train]
X_test = X.iloc[test,:]
y_test = y.iloc[test]

from sklearn.neighbors import KNeighborsRegressor

# Determine the optimal k:
k_bar = 100
k_grid = np.arange(1,k_bar) # The range of k's to consider
SSE = np.zeros(k_bar) 

for k in range(k_bar):
    knn = KNeighborsRegressor(n_neighbors=k+1)
    predictor = knn.fit(X_train,y_train) 
    y_hat = knn.predict(X_test)
    SSE[k] = np.sum( (y_test-y_hat)**2 ) # Bug in sklearn requires .values

SSE_min = np.min(SSE) # highest recorded accuracy
min_index = np.where(SSE==SSE_min) 
k_star = k_grid[min_index] # Find the optimal value of k
print(k_star)

plt.plot(np.arange(0,k_bar),SSE) # Plot accuracy by k
plt.xlabel("k")
plt.title("optimal k:"+str(k_star)+', SSE:'+str(SSE_min))
plt.ylabel('SSE')
plt.show()

# >> With more variables, the algorithm selects a higher optimal $k^* = 83$ instead of $k^*=6$, and it has a higher SSE of 29 rather than 27. The simpler model (fewer variables, fewer neighbors) does a better job predicting.
