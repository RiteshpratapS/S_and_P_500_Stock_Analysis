import pandas as pd

data = pd.read_csv('C:/Users/ritpr/Downloads/DOWNLOAD_1/CogniTensor Hiring Task/CTsp500project-py/Data/cs-1.csv')
data['date'] = pd.to_datetime(data['date'])
groups = data.groupby(data.date.dt.year)

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop



def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]



for group in groups:
    p = group[1].pivot(index='date', columns='Name', values='close')
    print("for year "+str(group[0]))
    print("Top 5 strongest pair :",get_top_abs_correlations(p, 5))

    