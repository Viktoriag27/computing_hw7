import pandas as pd
from sklearn.model_selection import train_test_split  # For splitting data

# Defining function to load csv file
def load_data(file_path):                  
    """
    Load a CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file to be loaded.
    
    Returns:
    pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(file_path)

# Function to clean and preprocess the data
def clean_data(df):
    """
    Clean and preprocess the DataFrame.
    - Remove rows with missing values in 'age', 'gender', or 'ethnicity'.
    - Fill missing 'height' and 'weight' values with the column mean.
    - Perform one-hot encoding on 'ethnicity'.
    - Map 'gender' values ('M' -> 1, 'F' -> 0).
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be cleaned.
    
    Returns:
    pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    # Drop rows with missing values in specific columns ('age', 'gender', 'ethnicity')
    df.dropna(subset=['age', 'gender', 'ethnicity'], inplace = True)
    
    # Fill missing values in 'height' and 'weight' with the respective column's mean
    df['height'].fillna(df['height'].mean(), inplace=True)
    df['weight'].fillna(df['weight'].mean(), inplace=True)
    
    # Convert 'ethnicity' column into one-hot encoded columns
    df = pd.get_dummies(df, columns=['ethnicity'])
    
    # Map 'gender' values ('M' -> 1, 'F' -> 0)
    df['gender'] = df['gender'].map({'M': 1, 'F': 0})
    
    return df  # Return the cleaned DataFrame

# Function to split the data into training and testing sets
def split_data(df):
    """
    Split the DataFrame into training and test sets.
    The target variable is 'diabetes_mellitus', and the features are other columns like 'age', 'height', etc.
    
    Parameters:
    df (pd.DataFrame): The cleaned and preprocessed DataFrame.
    
    Returns:
    X_train, X_test, y_train, y_test: Features and target variables for training and testing.
    """
    # Define the feature columns (X) and target column (y)
    X = df[['age', 'height', 'weight', 'aids', 'cirrhosis', 'hepatic_failure',
            'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']]
    y = df['diabetes_mellitus']  # Target column (diabetes presence)
    
    # Split the data into training (70%) and test (30%) sets, with a fixed random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test  # Return the split datasets