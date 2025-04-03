import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

def scrape_imdb_top_movies(url="https://www.imdb.com/chart/top/"):
    # Set a user agent to avoid being blocked
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Send request to IMDb
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all movie containers
    movie_containers = soup.select('.ipc-metadata-list-summary-item')
    
    movies_data = []
    
    # Extract data for each movie
    for i, movie in enumerate(movie_containers[:250]):  # Limit to 250 movies
        try:
            # Movie title
            title_elem = movie.select_one('.ipc-title__text')
            title = title_elem.text.split('. ')[1] if title_elem else "N/A"
            
            # Movie link for additional details
            movie_link = "https://www.imdb.com" + movie.select_one('a')['href'] if movie.select_one('a') else None
            
            # Rating
            rating = float(movie.select_one('.ipc-rating-star').text.split()[0]) if movie.select_one('.ipc-rating-star') else None
            
            # Year, runtime and other metadata
            metadata = movie.select('.cli-title-metadata-item')
            year = int(metadata[0].text) if len(metadata) > 0 else None
            runtime_text = metadata[1].text if len(metadata) > 1 else "0m"
            runtime = int(re.search(r'(\d+)h', runtime_text).group(1)) * 60 + int(re.search(r'(\d+)m', runtime_text).group(1)) if re.search(r'(\d+)h', runtime_text) and re.search(r'(\d+)m', runtime_text) else None
            
            # Store basic movie info
            movie_info = {
                'title': title,
                'year': year,
                'rating': rating,
                'runtime': runtime,
                'link': movie_link
            }
            
            # Add to our list
            movies_data.append(movie_info)
            
            print(f"Processed {i+1}/250: {title}")
            
        except Exception as e:
            print(f"Error processing movie: {e}")
    
    # Create a DataFrame
    movies_df = pd.DataFrame(movies_data)
    
    # Get additional details for each movie (votes and genres)
    movies_df = get_additional_details(movies_df)
    
    return movies_df

def get_additional_details(movies_df):
    """Get votes and genres for each movie by scraping individual movie pages"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Add new columns
    movies_df['votes'] = None
    movies_df['genres'] = None
    
    for idx, row in movies_df.iterrows():
        try:
            # Fetch movie page
            movie_url = row['link']
            response = requests.get(movie_url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract votes
                votes_elem = soup.select_one('[data-testid="rating-count"]')
                if votes_elem:
                    votes_text = votes_elem.text.replace(',', '').replace('K', '000')
                    votes = int(re.search(r'(\d+)', votes_text).group(1)) if re.search(r'(\d+)', votes_text) else None
                    movies_df.at[idx, 'votes'] = votes
                
                # Extract genres
                genres_elems = soup.select('a.ipc-chip--on-baseAlt span')
                genres = [elem.text for elem in genres_elems if elem.text not in ['Back to top', 'Add to watchlist']][:3]  # Get first 3 genres
                movies_df.at[idx, 'genres'] = ', '.join(genres) if genres else None
                
                print(f"Got details for: {row['title']}")
            
            # Delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error getting details for {row['title']}: {e}")
    
    return movies_df

if __name__ == "__main__":
    # Scrape top movies
    movies_df = scrape_imdb_top_movies()
    
    # Save data to CSV
    if movies_df is not None:
        movies_df.to_csv('imdb_top_movies.csv', index=False)
        print(f"Saved data for {len(movies_df)} movies to imdb_top_movies.csv")