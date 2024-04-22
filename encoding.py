import pandas as pd

def do_encoding():
    raw_df = pd.read_csv("./Breast_Cancer_dataset.csv")

    data_cleaned = raw_df.dropna()
    dropped_rows = raw_df.loc[raw_df.index.difference(data_cleaned.index)]
    print("Rows with missing values that were dropped:")
    print(dropped_rows)

    raw_df = data_cleaned
    
    raw_df['T Stage'] = raw_df['T Stage'].map({'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3})
    raw_df['N Stage'] = raw_df['N Stage'].map({'N1': 0, 'N2': 1, 'N3': 2})
    raw_df['A Stage'] = raw_df['A Stage'].map({'Regional': 0, 'Distant': 1})
    raw_df['6th Stage'] = raw_df['6th Stage'].map({'IIA': 0, 'IIB': 1, 'IIIA': 2, 'IIIB': 3, 'IIIC': 4})
    raw_df['Estrogen Status'] = raw_df['Estrogen Status'].map({'Positive': 0, 'Negative': 1})
    raw_df['Grade'] = raw_df['Grade'].map({'1': 0, '2': 1, '3': 2, '4': 3})
    raw_df['differentiate'] = raw_df['differentiate'].map({'Undifferentiated': 0, 'Poorly differentiated': 1, 'Moderately differentiated': 2, 'Well differentiated': 3})
    raw_df['Progesterone Status'] = raw_df['Progesterone Status'].map({'Positive': 1, 'Negative': 0})
    raw_df['Status'] = raw_df['Status'].map({'Alive': 1, 'Dead': 0})

    onehot_encoded_df = pd.get_dummies(raw_df[['Race', 'Marital Status']], drop_first=False)
    
    raw_df = pd.concat([raw_df, onehot_encoded_df], axis=1)
    
    raw_df.drop(['Race', 'Marital Status'], axis=1, inplace=True)

    raw_df.to_csv('./preprocessed/encoded_dataset.csv', index=False)