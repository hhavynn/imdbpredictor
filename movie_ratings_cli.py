import argparse
import pandas as pd
import os
from prediction_function import predict_rating, save_model
from data_preprocessing import preprocess_data
from model_training import train_and_evaluate_models

def run_scraper():
    """Run the IMDb scraper"""
    from imdb_scraper import scrape_imdb_top_movies
    print("Scraping IMDb top movies...")
    movies_df = scrape_imdb_top_movies()
    if movies_df is not None:
        movies_df.to_csv('imdb_top_movies.csv', index=False)
        print(f"Saved data for {len(movies_df)} movies to imdb_top_movies.csv")
    return movies_df is not None

def train_model():
    """Train the prediction model"""
    if not os.path.exists('imdb_top_movies.csv'):
        print("Error: Dataset not found. Please run scraping first.")
        return False
        
    print("Processing data...")
    X_train, X_test, y_train, y_test, scaler, top_genres = preprocess_data()
    
    print("Training models...")
    models, results, predictions, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    best_model = models[best_model_name]
    
    print(f"Saving best model ({best_model_name})...")
    save_model(best_model, scaler, top_genres)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="IMDb Movie Rating Predictor")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Scraper command
    scraper_parser = subparsers.add_parser('scrape', help='Scrape IMDb top movies data')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the prediction model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict movie rating')
    predict_parser.add_argument('--year', type=int, required=True, help='Release year')
    predict_parser.add_argument('--runtime', type=int, required=True, help='Runtime in minutes')
    predict_parser.add_argument('--votes', type=int, required=True, help='Number of votes')
    predict_parser.add_argument('--genres', type=str, required=True, 
                              help='Comma-separated list of genres (e.g., "Drama,Action,Comedy")')
    
    args = parser.parse_args()
    
    if args.command == 'scrape':
        run_scraper()
    
    elif args.command == 'train':
        train_model()
    
    elif args.command == 'predict':
        # Parse genres from comma-separated string
        genres = [genre.strip() for genre in args.genres.split(',')]
        
        # Make prediction
        predicted_rating = predict_rating(
            year=args.year,
            runtime_minutes=args.runtime,
            votes=args.votes,
            genres=genres
        )
        
        if predicted_rating is not None:
            print(f"\nMovie details:")
            print(f"- Year: {args.year}")
            print(f"- Runtime: {args.runtime} minutes")
            print(f"- Votes: {args.votes}")
            print(f"- Genres: {', '.join(genres)}")
            print(f"\nPredicted IMDb rating: {predicted_rating:.2f}/10")
        else:
            print("Error: Could not make prediction. Make sure you've trained the model first.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()