import pandas as pd

data = pd.read_csv('C:/Users/ritpr/Downloads/DOWNLOAD_1/CogniTensor Hiring Task/CTsp500project-py/Data/cs-1.csv')
data.drop(columns=[ 'open', 'high', 'low','volume'] , inplace=True)
data['date'] = pd.to_datetime(data['date'])
data['vola'] = data['close'].rolling(window=7).std()
print(data.head(10))