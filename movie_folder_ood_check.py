import os
import glob

def create_movie_episodes_list(root_data_dir):
    """
    Scan movie10 folder and create episode dictionaries for all movies
    
    Returns
    -------
    list
        List of episode dicts with type='movie'
    """
    movies = []
    movies_root = os.path.join(root_data_dir, "algonauts_2025.competitors", "testdata", "movies", "ood")
    
    if not os.path.exists(movies_root):
        print(f"  ⚠ Movies folder not found: {movies_root}")
        return movies
    
    # Expected movie genres
    genres = ['chaplin', 'mononoke', 'passepartout', 'planetearth', 'pulpfiction', 'wot']
    
    for genre in genres:
        genre_path = os.path.join(movies_root, genre)
        if not os.path.exists(genre_path):
            continue
        
        print(genre_path)

        # Find all .mkv files in the genre folder (e.g., task-chaplin_video.mkv)
        mkv_files = sorted(glob.glob(os.path.join(genre_path, "task-*_video.mkv")))
        
        for mkv_file in mkv_files:
            filename = os.path.basename(mkv_file).replace('.mkv', '')
            # Extract movie name from format: task-{movie}_video -> movie
            movie_name = filename.replace('task-', '').replace('_video', '')
            movies.append({
                'episode': filename,           # e.g., 'task-chaplin_video'
                'genre': genre,                # e.g., 'chaplin'
                'title': movie_name,           # Movie title
                'type': 'movie',               # Content type marker
                'duration': 1.49               # Standard duration for feature extraction (in minutes)
            })
    
    print(f"Found {len(movies)} movie files across {len([g for g in genres if os.path.exists(os.path.join(movies_root, g))])} genres")
    return movies

root_data_dir = r"D:\fmri-algonauts-2025-data"

create_movie_episodes_list(root_data_dir)