import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def do_reduction():
    data = pd.read_csv('./preprocessed/standardized_dataset.csv')

    data = data.dropna()
    data = data.reset_index(drop=True)

    numerical_columns = [col for col in data.columns if col != 'Status']

    label = data['Status']
    data_features = data.drop(columns=['Status'])

    scaler = StandardScaler()   

    data_features[numerical_columns] = scaler.fit_transform(data_features[numerical_columns])

    pca = PCA(n_components=12)
    principal_components = pca.fit_transform(data_features[numerical_columns])

    principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

    final_data = pd.concat([principal_df, label], axis=1)

    final_data.to_csv('./preprocessed/reduced_dataset.csv', index=False)