import pandas as pd
import matplotlib as plt
import numpy as np

def missing_features(df):
    mask = df.isnull() 
    total = mask.sum()
    percent = 100*mask.mean()
    missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['count_missing', 'perc_missing'])
    
    ### get names of indexes for which count_missing value is 0
    index_names = missing_data[missing_data['count_missing']== 0].index 
    missing_data.drop(index_names, inplace = True)
    
    missing_data.sort_values(by='perc_missing', ascending=False, inplace=True)
    return missing_data

def uniqueness(df):
    column= []
    unique_number = []
    for col in df.columns:
        if df[col].dtypes == 'object':
            column.append(col)
            unique_number.append(df[col].nunique())
    return pd.DataFrame({'Features':column, 'No_of_uniqueness': unique_number}, index = range(len(column)) )

def reshape_columns(df):
    col_names = [col.lower().replace(' ', '_') for col in df.columns]
    df.columns = col_names
    return df

def check_for_skew_kurtosis(df, num_features):
    num_cols = num_features.columns
    skew = []
    kurt = []
    symmetry = []
    normal_data = []
    skewed_num_data = []
    for col in num_features.columns:
        x=num_features[col].skew()
        y = num_features[col].kurt()
        skew.append(x)
        kurt.append(y)
        if x < -0.5:
            symmetry.append('Left Skewed')
            skewed_num_data.append(col)
        elif x > 0.5:
            symmetry.append('Right Skewed')
            skewed_num_data.append(col)
        else:
            symmetry.append('Approximate Symmetry')
            normal_data.append(col)
    return pd.DataFrame({'Skew':skew, 'Kurtosis': kurt, 'Symmetry':symmetry}, index = num_cols )



def find_outlier_z_score(x):
    mean = np.mean(x)
    std = np.std(x)
    cut_off = std * 3
    lower, upper = mean - cut_off, mean + cut_off
    outlier_indices = list(x.index[(x < lower) | (x > upper)])
    outlier_values = list(x[outlier_indices])
    return outlier_indices, outlier_values

def find_outlier_IQR(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    IQR = q3-q1
    lower = q1 - 1.5*IQR
    upper =  q1 + 1.5*IQR
    outlier_indices = list(x.index[(x < lower) | (x > upper)])
    outlier_values = list(x[outlier_indices])
    return outlier_indices, outlier_values

def plot_importance(model,data,top_feat):
    """
    Plot feature importance for trained models.
    parameters::
    model: pre-trained estimator,
    data: data which estimator was trained on
    top_feat: how many top features do you want displayed??
    """
    fea_imp = pd.DataFrame({'imp':model.feature_importances_, 'col': data.columns})
    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-top_feat:]
    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
    plt.savefig('catboost_feature_importance.png')

