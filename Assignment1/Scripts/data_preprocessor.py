# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    data_copy = data.copy()
    if strategy == 'mean':
        for col in data_copy.select_dtypes(include='number').columns:
            data_copy[col].fillna(data_copy[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in data_copy.select_dtypes(include='number').columns:
            data_copy[col].fillna(data_copy[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in data_copy.columns:
            mode_val = data_copy[col].mode()
            if not mode_val.empty:
                data_copy[col].fillna(mode_val[0], inplace=True)
    else:
        raise ValueError("Invalid strategy. Choose 'mean', 'median', or 'mode'.")
    return data_copy

#1.2 new_version_impute

def impute_missing_values_new(data, recurrent_threshold=0.2, unique_value_cap=10):
    """
    Imputes missing values based on data type and distribution.
    - Continuous numeric: mean (or median if skewed)
    - Recurrent numeric (few unique values): mode
    - Categorical/text: mode

    :param data: pandas DataFrame
    :param recurrent_threshold: float, max proportion of unique values to consider recurrent
    :param unique_value_cap: int, optional cap for recurrent numeric values
    :return: pandas DataFrame with imputed values
    """
    data_copy = data.copy()

    for col in data_copy.columns:
        if data_copy[col].isnull().sum() == 0:
            continue  # skip columns without missing values

        if pd.api.types.is_numeric_dtype(data_copy[col]):
            unique_vals = data_copy[col].dropna().unique()
            n_unique = len(unique_vals)
            total = data_copy[col].dropna().shape[0]

            # Check for binary or recurrent numeric
            if n_unique <= unique_value_cap or n_unique / total <= recurrent_threshold:
                strategy = 'mode'
            else:
                skewness = data_copy[col].skew()
                strategy = 'median' if abs(skewness) > 1 else 'mean'

        else:
            # For categorical/text/boolean
            strategy = 'mode'

        if strategy == 'mean':
            impute_val = data_copy[col].mean()
        elif strategy == 'median':
            impute_val = data_copy[col].median()
        elif strategy == 'mode':
            mode_val = data_copy[col].mode()
            impute_val = mode_val[0] if not mode_val.empty else None
        else:
            raise ValueError(f"Unhandled strategy for column {col}")

        if impute_val is not None:
            data_copy[col].fillna(impute_val, inplace=True)

    return data_copy



# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    return data.copy().drop_duplicates()

# 3. Normalize Numerical Data
def normalize_data(data, method='minmax'):
    """
    Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' or 'standard')
    :return: pandas DataFrame
    """
    data_copy = data.copy()
    num_cols = data_copy.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid method. Choose 'minmax' or 'standard'.")
    
    if len(num_cols) > 0:
        data_copy[num_cols] = scaler.fit_transform(data_copy[num_cols])
    return data_copy

#3.2 

def normalize_data_new(data, skew_threshold=1.0):
    """
    Automatically normalize numeric columns using MinMaxScaler or StandardScaler
    based on skewness of the data.

    :param data: pandas DataFrame
    :param skew_threshold: float, threshold to decide if data is skewed
    :return: pandas DataFrame with normalized numeric columns
    """
    data_copy = data.copy()
    num_cols = data_copy.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        if data_copy[col].isnull().sum() > 0:
            continue  # Skip columns with missing values (or impute before this)

        skewness = data_copy[col].skew()
        if abs(skewness) > skew_threshold:
            scaler = MinMaxScaler()
            method = 'minmax'
        else:
            scaler = StandardScaler()
            method = 'standard'

        # Reshape is needed for single-column scaling
        reshaped = data_copy[col].values.reshape(-1, 1)
        data_copy[col] = scaler.fit_transform(reshaped)

    return data_copy

# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """
    Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    data_copy = data.copy()
    num_cols = data_copy.select_dtypes(include=[np.number]).columns
    
    
    corr_matrix = data_copy[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return data_copy.drop(columns=to_drop)



# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None

def simple_model_new(input_data, split_data=True, scale_data=False, print_report=False, target_bins=None):
    """
    A simple logistic regression model for target classification.
    """
    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    # Convert continuous target to categorical
    if target_bins is not None:
        #Use custom bins
        target = pd.cut(target, bins=target_bins, labels=False)
    else:
        #Auto-create bins (e.g., low, medium, high)
        target = pd.cut(target, bins=3, labels=['Low', 'Medium', 'High'])
    
    print(f"Target distribution after binning:\n{target.value_counts()}")

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    if print_report:
        print('Classification Report:')
        print(report)
    
    return None