#!/usr/bin/env python3
"""
Movie Ratings Predictor Using IMDb Data
--------------------------------------
This project scrapes movie data from IMDb, preprocesses it, and builds 
a regression model to predict IMDb ratings based on movie attributes.
"""

import os
import sys

def print_header():
    print("""
    ╔════════════════════════════════════════════════════╗
    ║                                                    ║
    ║           IMDb Movie Ratings Predictor             ║
    ║                                                    ║
    ╚════════════════════════════════════════════════════╝
    """)

def show_usage():
    print("""
    Usage: python main.py [command]
    
    Available commands:
      setup     - Run the complete setup (scrape data and train model)
      scrape    - Only scrape IMDb data
      train     - Only train the prediction model
      predict   - Make predictions for a new movie
      help      - Show this help message
    """)

def setup():
    """Setup the entire project (scrape and train)"""
    from movie_ratings_cli import run_scraper, train_model
    
    print_header()
    print("\n[1/2] Setting up the project...\n")
    
    # Step 1: Scrape data
    print("Step 1: Scraping IMDb data")
    if not run_scraper():
        print("Error during scraping. Aborting setup.")
        return False
    
    # Step 2: Train model
    print("\nStep 2: Training prediction model")
    if not train_model():
        print("Error during model training. Aborting setup.")
        return False
    
    print("\n✓ Setup completed successfully!")
    print("\nYou can now use 'python main.py predict' to make predictions.")
    return True

def interactive_predict():
    """Interactive prediction interface"""
    from prediction_function import predict_rating
    
    print_header()
    print("\nEnter movie details to predict IMDb rating:\n")
    
    try:
        year = int(input("Release year: "))
        runtime = int(input("Runtime (minutes): "))
        votes = int(input("Number of votes: "))
        genres_input = input("Genres (comma-separated, e.g., Drama,Action,Adventure): ")
        genres = [genre.strip() for genre in genres_input.split(',')]
        
        predicted_rating = predict_rating(
            year=year,
            runtime_minutes=runtime,
            votes=votes,
            genres=genres
        )
        
        if predicted_rating is not None:
            print(f"\nMovie details:")
            print(f"- Year: {year}")
            print(f"- Runtime: {runtime} minutes")
            print(f"- Votes: {votes}")
            print(f"- Genres: {', '.join(genres)}")
            print(f"\nPredicted IMDb rating: {predicted_rating:.2f}/10")
        else:
            print("\nError: Could not make prediction. Make sure you've trained the model first.")
    
    except ValueError:
        print("Invalid input. Please enter numeric values for year, runtime, and votes.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    # Get command line arguments
    if len(sys.argv) < 2:
        command = "help"
    else:
        command = sys.argv[1].lower()
    
    # Process command
    if command == "setup":
        setup()
    
    elif command == "scrape":
        from movie_ratings_cli import run_scraper
        print_header()
        run_scraper()
    
    elif command == "train":
        from movie_ratings_cli import train_model
        print_header()
        train_model()
    
    elif command == "predict":
        interactive_predict()
    
    elif command in ["help", "-h", "--help"]:
        print_header()
        show_usage()
    
    else:
        print(f"Unknown command: {command}")
        show_usage()

if __name__ == "__main__":
    main()