import pandas as pd
from sklearn.preprocessing import StandardScaler

def do_normalization():
    data = pd.read_csv('./preprocessed/cleaned_dataset.csv')

    numerical_columns = [col for col in data.columns if col != 'Status']

    label = data['Status']
    data_features = data.drop(columns=['Status'])

    scaler = StandardScaler()

    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    final_data = pd.concat([data, label], axis=1)
    final_data.to_csv('./preprocessed/standardized_dataset.csv', index=False)
