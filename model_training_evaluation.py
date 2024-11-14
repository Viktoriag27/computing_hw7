from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# train_model: Function to train the Logistic Regression model
def train_model(X_train, y_train):
    """
    Train a Logistic Regression model using the training data.
    
    Parameters:
    X_train (pd.DataFrame): The feature set for training.
    y_train (pd.Series): The target variable for training.
    
    Returns:
    model (LogisticRegression): The trained Logistic Regression model.
    """
    # Initialize the Logistic Regression model
    model = LogisticRegression()
    
    # Fit the model using the training data (X_train and y_train)
    model.fit(X_train, y_train)
    
    # Return the trained model
    return model

# evaluate_model: Function to make predictions and compute ROC AUC score
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate the Logistic Regression model by predicting probabilities and computing the ROC AUC score.
    
    Parameters:
    model (LogisticRegression): The trained Logistic Regression model.
    X_train (pd.DataFrame): The feature set for training.
    X_test (pd.DataFrame): The feature set for testing.
    y_train (pd.Series): The target variable for training.
    y_test (pd.Series): The target variable for testing.
    
    Returns:
    train_roc_auc (float): The ROC AUC score for the training set predictions.
    test_roc_auc (float): The ROC AUC score for the test set predictions.
    """
    # Predict the probabilities for the training set (we take the probabilities for the positive class)
    y_train_pred = model.predict_proba(X_train)[:, 1]
    
    # Predict the probabilities for the test set (positive class probabilities)
    y_test_pred = model.predict_proba(X_test)[:, 1]
    
    # Compute ROC AUC score for training set predictions
    train_roc_auc = roc_auc_score(y_train, y_train_pred)
    
    # Compute ROC AUC score for test set predictions
    test_roc_auc = roc_auc_score(y_test, y_test_pred)
    
    # Return both ROC AUC scores (train and test)
    return train_roc_auc, test_roc_auc