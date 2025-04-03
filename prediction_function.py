import pandas as pd
import numpy as np
import pickle
import os
from data_preprocessing import preprocess_data
from model_training import train_and_evaluate_models

def save_model(model, scaler, top_genres, model_name="best_model"):
    """Save the trained model and its dependencies"""
    model_info = {
        'model': model,
        'scaler': scaler,
        'top_genres': top_genres
    }
    
    with open(f'{model_name}.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"Model saved as {model_name}.pkl")

def load_model(model_path="best_model.pkl"):
    """Load the trained model and its dependencies"""
    try:
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        
        return model_info['model'], model_info['scaler'], model_info['top_genres']
    except:
        print(f"Error loading model from {model_path}")
        return None, None, None

def predict_rating(year, runtime_minutes, votes, genres, model_path="best_model.pkl"):
    """
    Predict IMDb rating for a new movie
    
    Parameters:
    -----------
    year : int
        Release year of the movie
    runtime_minutes : int
        Runtime in minutes
    votes : int
        Number of votes on IMDb
    genres : list
        List of genres
    model_path : str
        Path to the saved model file
    
    Returns:
    --------
    float
        Predicted IMDb rating
    """
    # Load the model and its dependencies
    model, scaler, top_genres = load_model(model_path)
    
    if model is None:
        return None
    
    # Prepare input data
    decade = (year // 10) * 10
    log_votes = np.log1p(votes)
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'year': [year],
        'runtime': [runtime_minutes],
        'log_votes': [log_votes],
        'decade': [decade]
    })
    
    # Add genre columns (one-hot encoding)
    for genre in top_genres:
        input_data[genre] = 1 if genre in genres else 0
    
    # Scale numeric features
    numeric_cols = ['year', 'runtime', 'log_votes', 'decade']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return prediction

def main():
    # Check if model exists, if not, train it
    if not os.path.exists('best_model.pkl'):
        print("Training new model...")
        X_train, X_test, y_train, y_test, scaler, top_genres = preprocess_data()
        models, results, predictions, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        best_model = models[best_model_name]
        save_model(best_model, scaler, top_genres)
    
    # Example prediction
    print("\nExample Prediction:")
    predicted_rating = predict_rating(
        year=2023,
        runtime_minutes=140,
        votes=50000,
        genres=['Drama', 'Action', 'Sci-Fi']
    )
    
    print(f"Predicted IMDb rating: {predicted_rating:.2f}/10")

if __name__ == "__main__":
    main()