import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(csv_path='imdb_top_movies.csv'):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Display basic info
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Alternative approach for missing votes
    # Instead of dropping rows, let's fill missing votes with a reasonable value
    # For top movies with missing votes, we can use the median of available votes or a high constant
    if 'votes' in df.columns and df['votes'].isnull().any():
        # Check if we have any valid votes to calculate median
        if df['votes'].notnull().any():
            median_votes = df['votes'].median()
            df['votes'] = df['votes'].fillna(median_votes)
            print(f"\nFilled missing votes with median: {median_votes}")
        else:
            # If all votes are missing, use a reasonable default for popular movies
            default_votes = 500000  # A reasonable number for top movies
            df['votes'] = df['votes'].fillna(default_votes)
            print(f"\nAll votes were missing. Using default value: {default_votes}")
    
    # Handle any remaining missing values in critical columns
    # We'll only drop if absolutely necessary
    critical_cols = ['rating', 'year', 'runtime']
    missing_critical = df[critical_cols].isnull().any(axis=1)
    if missing_critical.any():
        print(f"\nDropping {missing_critical.sum()} rows with missing critical data (rating, year, runtime)")
        df = df.dropna(subset=critical_cols)
    
    # Fill missing genres with 'Unknown'
    df['genres'] = df['genres'].fillna('Unknown')
    
    # Feature engineering
    
    # 1. Extract decade from year
    df['decade'] = (df['year'] // 10) * 10
    
    # 2. Log transform votes (because votes typically follow a power law distribution)
    df['log_votes'] = np.log1p(df['votes'])
    
    # 3. Process genres
    # Split genres into individual ones and create binary columns
    genre_dummies = df['genres'].str.get_dummies(sep=', ')
    
    # Keep only the top N most common genres to avoid having too many features
    # Ensure we have at least some genres to work with
    if genre_dummies.shape[1] > 0:
        # Take top 10 or all if less than 10
        n_genres = min(10, genre_dummies.shape[1])
        top_genres = genre_dummies.sum().sort_values(ascending=False).head(n_genres).index
        genre_dummies = genre_dummies[top_genres]
    else:
        # If no genre data is available, create a dummy genre
        genre_dummies = pd.DataFrame({'Unknown': [1] * len(df)}, index=df.index)
        top_genres = ['Unknown']
    
    # Add genre dummies to dataframe
    df = pd.concat([df, genre_dummies], axis=1)
    
    # Prepare features and target
    X = df[['year', 'runtime', 'log_votes', 'decade'] + list(top_genres)]
    y = df['rating']
    
    # Ensure we have enough data for splitting
    if len(df) < 5:
        print("\nWARNING: Very small dataset detected. Using all data for training.")
        # Return the same data for both train and test when dataset is too small
        return X, X.copy(), y, y.copy(), StandardScaler().fit(X[['year', 'runtime', 'log_votes', 'decade']]), top_genres
    
    # Split data into training and testing sets
    test_size = min(0.2, 1/len(df) * 2)  # Ensure at least 1 sample in test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Feature scaling for numeric columns
    numeric_cols = ['year', 'runtime', 'log_votes', 'decade']
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print("\nPreprocessing complete!")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.columns.tolist()}")
    
    return X_train, X_test, y_train, y_test, scaler, top_genres

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, top_genres = preprocess_data()