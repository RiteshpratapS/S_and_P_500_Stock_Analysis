import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv('C:/Users/ritpr/Downloads/DOWNLOAD_1/CogniTensor Hiring Task/CTsp500project-py/Data/cs-1.csv')
print(data.head())
# data=data.drop(['volume'], axis=1)


# calculate momentum for each day
# 5-day momentum
def momentum(df):
    n = len(df)
    arr = []
    for i in range(0,5):
        arr.append('N')
    for j in range(5,n):
        momentum = df.close[j] - df.close[j-5] #Equation for momentum
        arr.append(momentum)
    return arr


momentum = momentum(data)

# add momentum to data
data['Momentum'] = momentum

#Use pct_change() function to add the one day returns to the dataframe 
data_pctchange=data.close.pct_change()
data['Return'] = data_pctchange

#Function to Classify each day as a 1 or a 0
def clas(df):
    n = len(df)
    arr = []
    for i in range(0,len(df)-1):
        if (100*((df.close[i+1]-df.open[i+1])/df.open[i+1]))>=.3:
            arr.append(1)
        else:
            arr.append(0)
    arr.append('N')
    return arr


clas=clas(data)

#Add Class to our dataframe
data['Class'] = clas

# calculate Commodity Channel Index (CCI) for each day
import numpy as np
def CCI(df,n):
    m = len(df)
    arr = []
    tparr = []
    for i in range(0,n-1):
        arr.append('N')
        tp = (df.high[i]+df.low[i]+df.close[i])/3
        tparr.append(tp)
    for j in range(n-1,m):
        tp = (df.high[j]+df.low[j]+df.close[j])/3
        tparr.append(tp) 
        tps = np.array(tparr[(j-n+1):(j+1)])
        val = (tp-tps.mean())/(0.015*tps.std())
        arr.append(val)
    return arr


cci = CCI(data,20) 

#Add CCI to our dataframe
data['CCI'] = cci

#double check that the dataframe has all 10 features
data.shape

#def normalization function to clean data
def normalize(df):
    for column in df:
        df[column]=((df[column]-df[column].mean())/df[column].std())


#def positive values for running Multinomial Naive Bayes
def positivevalues(df):
    for column in df:
        if (df[column].min())<0:
            df[column]=(df[column]-df[column].min())


#Remove the first 30 index which could have a value 'N'
newdata=data.drop(data.index[0:30])

#Remove the last row of data because class has value 'N'
newdata=newdata.drop(newdata.index[-1])

#Remove 'High' and 'Low' columns to improve the algorithm
newdata=newdata.drop(['high','low'], axis=1)

#check the features that remain in our algorithm 
print(newdata.head())

newdata = newdata.drop(['Name', 'date'], axis=1)

#Put the dataframe with our relevant features into X and our class into our y
X=newdata
y=clas[30:-1]

#Split up our test and train by splitting 70%/30%
X_train=X.drop(X.index[1211:])
X_test=X.drop(X.index[0:1211])
y_train=y[0:1211]
y_test=y[1211:]

#Import and run Logistic Regression and run a fit to train the model
LR=LogisticRegression()
LR.fit(X_train,y_train)

X_test = X_test.fillna(0)

#Predict the y test
y_pred_LR=LR.predict(X_test)

#Print the accuracy score of our predicted y using metrics from sklearn
print (metrics.accuracy_score(y_test, y_pred_LR))

#Import and run Gaussian Naive Bayes and run a fit to train the model
GNB = GaussianNB()
GNB.fit(X_train,y_train)

#Predict the y test
y_pred=GNB.predict(X_test)

#Print the accuracy score of our predicted y using metrics from sklearn
print (metrics.accuracy_score(y_test, y_pred))


'''
#def get_pred(test_data, test_label):
    momentum = momentum(test_data)
    test_data['Momentum'] = momentum
    data_pctchange=test_data.close.pct_change()
    test_data['Return'] = data_pctchange
    clas=clas(test_data)
    data['Class'] = clas
    cci = CCI(data,20)
    test_data['CCI'] = cci
    newdata=test_data.drop(data.index[0:30])
    newdata=newdata.drop(['high','low'], axis=1)
    newdata = newdata.drop(['Name', 'date'], axis=1)
    y_pred_LR=LR.predict(newdata)
    acc=metrics.accuracy_score(test_label, y_pred_LR)

    return acc

get_pred(X_test,y_test)

 '''