import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('C:/Users/ritpr/Downloads/DOWNLOAD_1/CogniTensor Hiring Task/CTsp500project-py/Data/cs-1.csv')
print(data.head())
data=data.drop(['volume'], axis=1)


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

#ROI function

def ROI(df,n):
    m = len(df)
    arr = []
    for i in range(0,n):
        arr.append('N')
    for j in range(n,m):
        roi= (df.close[j] - df.close[j-n])/df.close[j-n] #Equation for ROI
        arr.append(roi)
    return arr

#Run the ROI function for 10, 20, and 30 day periods

ROI10=ROI(data,10)
ROI20=ROI(data,20)
ROI30=ROI(data,30)


#Add all 3 ROI results to dataframe 

data['10 Day ROI']=ROI10
data['20 Day ROI']=ROI20
data['30 Day ROI']=ROI30


# calculate RSI for each day


def RSI(df,period):
    # get average of upwards of last 14 days: Ct - Ct-1
    # get average of downwards of last 14 days: Ct-1 - Ct
    n = len(df)
    arr = []
    for i in range(0,period):
        arr.append('N')
    for j in range(period,n):
        total_upwards = 0
        total_downwards = 0
        # this will find average of upwards
        for k in range(j,j-period,-1):
            if(df.close[k-1] > df.close[k]):
                total_downwards = total_downwards + (df.close[k-1] - df.close[k])    
        avg_down = total_downwards / period
        for l in range(j,j-period,-1):
            if(df.close[l] > df.close[l-1]):
                total_upwards = total_upwards + (df.close[l] - df.close[l-1])
        avg_up = total_upwards / period
        RS = avg_up / avg_down
        RSI  = 100 - (100/(1+RS))
        arr.append(RSI)
    return arr


#Run RSI for 10, 14, and 30 day periods

RSI_14 = RSI(data,14)
RSI_10 = RSI(data,10)
RSI_30 = RSI(data,30)

# add RSI to data

data['10_day_RSI'] = RSI_10
data['14_day_RSI'] = RSI_14
data['30_day_RSI'] = RSI_30

# calculate EMA for each day
# formula: EMA = (2/(n+1))*ClosePrice + (1-(2/(n+1)))*previousEMA

def EMA(df, n):
    m = len(df)
    arr = []
    arr.append('N')
    prevEMA = df.close[0]
    for i in range(1,m):
        close = df.close[i]
        EMA = ((2/(n+1))*close) + ((1-(2/(n+1)))*prevEMA)
        arr.append(EMA)
        prevEMA = EMA
    return arr

#Calculate EMA with n=12 and n=26

EMA_12 = EMA(data, 12)
EMA_26 = EMA(data, 26)

#add EMA to dataframe 

data['EMA_12'] = EMA_12
data['EMA_26'] = EMA_26


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


#MACD
# Moving Average of EMA(n) - EMA(m2) for each row
# where n = 12 and m2 = 26
def MACD(df):
    n = 12
    m2 = 26
    arr = []
    arr.append('N')
    ema_12 = EMA(df,n)
    ema_26 = EMA(df,m2)
    m = len(df)
    for i in range(1,m):
        arr.append(ema_12[i] - ema_26[i])
    return arr

MACD = MACD(data)

#Add MACD to our dataframe 
data['MACD_12_26'] = MACD

#SRSI: Stochastic RSI
#SRSI = (RSI_today - min(RSI_past_n)) / (max(RSI_past_n) - min(RSI_past_n))
def SRSI(df,n):
    m = len(df)
    arr = []
    list_RSI = RSI(df,n)
    for i in range(0,n):
        arr.append('N')
    for j in range(n,n+n):
        last_n = list_RSI[n:j]
        if(not(last_n == []) and not(max(last_n) == min(last_n))):
            SRSI = (list_RSI[j] - min(last_n)) / (max(last_n)- min(last_n))
            if SRSI > 1:
                arr.append(1)
            else:
                arr.append(SRSI)
        else:
            arr.append(0)
    for j in range(n+n,m):
        last_n = list_RSI[2*n:j]
        if(not(last_n == []) and not(max(last_n) == min(last_n))):
            SRSI = (list_RSI[j] - min(last_n)) / (max(last_n)- min(last_n))
            if SRSI > 1:
                arr.append(1)
            else:
                arr.append(SRSI)
        else:
            arr.append(0)
    return arr

#Run SRSI for 10, 14, and 30 day periods
SRSI_10 = SRSI(data,10)
SRSI_14 = SRSI(data,14)
SRSI_30 = SRSI(data,30)

#Add SRSI to our dataframe
data['SRSI_10'] = SRSI_10
data['SRSI_14'] = SRSI_14
data['SRSI_30'] = SRSI_30

# calculate Williams %R oscillator for each day

def Williams(df,n):
    m = len(df)
    arr = []
    for i in range(0,n-1):
        arr.append('N')
    for j in range(n-1,m):
        maximum = max(data.high[(j-n+1):j+1])
        minimum = min(data.low[(j-n+1):j+1])
        val = (-100)*(maximum-df.close[j])/(maximum-minimum)
        arr.append(val)
    return arr


williams = Williams(data,14)

#Add Williams%R to our dataframe
data['Williams'] = williams

# calculate Williams %R oscillator for each day

def Williams(df,n):
    m = len(df)
    arr = []
    for i in range(0,n-1):
        arr.append('N')
    for j in range(n-1,m):
        maximum = max(data.high[(j-n+1):j+1])
        minimum = min(data.low[(j-n+1):j+1])
        val = (-100)*(maximum-df.close[j])/(maximum-minimum)
        arr.append(val)
    return arr


williams = Williams(data,14)

#Add Williams%R to our dataframe
data['Williams'] = williams

# True Range
# TR = MAX(high[today] - close[yesterday]) - MIN(low[today] - close[yesterday])
def TR(df,n):
    high = df.high[n]
    low = df.low[n]
    close = df.close[n-1]
    l_max = list()
    l_max.append(high)
    l_max.append(close)
    l_min = list()
    l_min.append(low)
    l_min.append(close)
    return (max(l_max) - min(l_min))

# Average True Range
# Same as EMA except use TR in lieu of close (prevEMA = TR(dataframe,14days))
def ATR(df,n):
    m = len(df)
    arr = []
    prevEMA = TR(df,n+1)
    for i in range(0,n):
        arr.append('N')
    for j in range(n,m):
        TR_ = TR(df,j)
        EMA = ((2/(n+1))*TR_) + ((1-(2/(n+1)))*prevEMA)
        arr.append(EMA)
        prevEMA = EMA
    return arr

ATR = ATR(data,14)  

#Add ATR to our dataframe
data['ATR_14'] = ATR


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

#double check that the dataframe has all 22 features
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

#Remove 'high' and 'low' columns to improve the algorithm
newdata=newdata.drop(['high','low'], axis=1)

#Remove our 'Class' column because it acts as y in our algorithms 
newdata=newdata.drop(['Class'], axis=1)

#check the features that remain in our algorithm 
newdata.head()

#Normalize the data that we have filtered
normalize(newdata)

#Put the dataframe with our relevant features into X and our class into our y
X=newdata
y=clas[30:-1]


#Split up our test and train by splitting 70%/30%

X_train=X.drop(X.index[1211:])
X_test=X.drop(X.index[0:1211])
y_train=y[0:1211]
y_test=y[1211:]


#Run Logistic Regression and run a fit to train the model
LR=LogisticRegression()
LR.fit(X_train,y_train)

#Predict the y test 
y_pred_LR=LR.predict(X_test)


#Print the accuracy score of our predicted y using metrics from sklearn
print (metrics.accuracy_score(y_test, y_pred_LR))

#Run Gaussian Naive Bayes and run a fit to train the model
GNB = GaussianNB()
GNB.fit(X_train,y_train)

#Predict the y test
y_pred=GNB.predict(X_test)

        