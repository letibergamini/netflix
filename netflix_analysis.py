import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Data Loading, Cleaning, and Analysis Functions ---

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the Netflix dataset."""
    return pd.read_csv(filepath)

def get_initial_overview(df: pd.DataFrame):
    """Provides an initial overview of the dataset."""
    print("--- Dataset Head ---\n", df.head())
    print("\n--- Dataset Tail ---\n", df.tail())
    print("\n--- Column Names ---\n", list(df.columns))
    print("\n--- Dataset Info ---")
    df.info()

def clean_and_extract_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Adjusts 'date_added' column and extracts 'year_added' and 'month_added'."""
    df["date_added"] = df["date_added"].str.strip()
    df["date_added"] = pd.to_datetime(df["date_added"])
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month
    return df

def clean_duration_and_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Separates 'duration' into 'duration_minutes' for movies and 'seasons' for TV shows."""
    df["duration_minutes"] = df["duration"].apply(lambda x: int(x.split(" ")[0]) if "min" in x else None)
    df["seasons"] = df["duration"].apply(lambda x: int(x.split(" ")[0]) if "min" not in x else None)
    df = df.drop(columns=['duration'])
    return df

def get_null_value_summary(df: pd.DataFrame):
    """Calculates and prints the count and share of null values for each column."""
    print(f"\n--- Null Values Per Variable ---\n {df.isnull().sum()}")
    for column in df.columns:
        null_share = ((df[column].isnull().sum()/df.shape[0]) * 100).round(2)
        print(f"Share of null values for variable {column} on total sample: {null_share}%")

def get_content_type_data(df: pd.DataFrame) -> pd.Series:
    """Calculates the counts of Movies and TV Shows."""
    print(f"\n--- Null values in 'type' column ---\n {df['type'].isnull().sum()}")
    type_counts = df.groupby("type").size()
    print("\n--- Content Type Counts ---\n", type_counts)
    type_share = (type_counts/(df.shape[0])*100).round(2)
    print("\n--- Content Type Share ---\n", type_share)
    return type_counts

def get_yearly_content_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares data for yearly content trends."""
    df_sorted = df.sort_values(by=["year_added", "month_added"], ascending=True)
    df_clean = df_sorted.dropna(subset=["year_added", "month_added"]).copy()
    df_clean["year_added"] = df_clean["year_added"].astype(int)
    df_clean["month_added"] = df_clean["month_added"].astype(int)
    
    type_added_by_year = df_clean.groupby(["year_added", "type"]).size().unstack(fill_value=0)
    print("\n--- Content Added by Year ---\n", type_added_by_year)
    return type_added_by_year

def get_monthly_content_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares data for monthly content trends."""
    df_clean = df.dropna(subset=["year_added", "month_added"]).copy()
    df_clean["month_added"] = df_clean["month_added"].astype(int)
    type_added_by_month = df_clean.groupby(["month_added", "type"]).size().unstack(fill_value=0)
    print("\n--- Content Added by Month ---\n", type_added_by_month)
    return type_added_by_month

def get_movie_duration_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts movie durations."""
    movies_duration_df = df[df["type"] == "Movie"][["type", "duration_minutes"]].dropna()
    print(f"\n--- Null values in movie durations: {movies_duration_df['duration_minutes'].isnull().sum()}")
    return movies_duration_df

def get_tv_show_season_data(df: pd.DataFrame) -> pd.Series:
    """Prepares TV show season distribution data."""
    shows_seasons_df = df[df["type"] == "TV Show"][["type", "seasons"]].dropna().copy()
    shows_seasons_df["seasons"] = shows_seasons_df["seasons"].astype(int)

    shows_seasons_df["season_range"] = shows_seasons_df["seasons"].apply(
        lambda x: "1-2 seasons" if x < 3 else
                   "3-5 seasons" if x > 2 and x < 6 else
                   "6-10 seasons" if x > 5 and x < 11 else
                   "11+ seasons"
    )
    season_order = ["1-2 seasons", "3-5 seasons", "6-10 seasons", "11+ seasons"]
    shows_seasons_df["season_range"] = pd.Categorical(
        shows_seasons_df["season_range"],
        categories=season_order,
        ordered=True
    )
    shows_by_seasons = shows_seasons_df.groupby("season_range").size()
    print("\n--- TV Shows by Season Range ---\n", shows_by_seasons)
    return shows_by_seasons

def get_oldest_content_data(df: pd.DataFrame):
    """Retrieves the oldest movies and TV shows."""
    movies_df = df[df["type"] == "Movie"]
    oldest_movies = movies_df.sort_values(["release_year", "title"])
    print(f"\n--- Oldest Movies on Netflix ---\n {oldest_movies[['show_id', 'title', 'type','release_year']].head(15)}")

    tvshows_df = df[df["type"] == "TV Show"]
    oldest_tvshows = tvshows_df.sort_values(["release_year", "title"])
    print(f"\n--- Oldest TV Shows on Netflix ---\n {oldest_tvshows[['show_id', 'title', 'type','release_year']].head(15)}")

def get_popular_directors_data(df: pd.DataFrame) -> pd.Series:
    """Extracts data for the most popular directors."""
    movies_director_clean = df.dropna(subset="director").copy()
    print(f"\n--- Null values in 'director' after dropping ---\n {movies_director_clean['director'].isnull().sum()}")
    most_popular_directors = movies_director_clean["director"].value_counts().head(10)
    print(f"\n--- 10 Most Popular Directors on Netflix ---\n {most_popular_directors}")
    return most_popular_directors

def get_popular_actors_data(df: pd.DataFrame) -> pd.Series:
    """Extracts data for the most popular actors."""
    print(f"\n--- Null values in 'cast' ---\n {df['cast'].isnull().sum()}")
    movies_actors_clean = df.dropna(subset="cast").copy()
    print(f"Null values in 'cast' after dropping: {movies_actors_clean['cast'].isnull().sum()}")

    movies_actors_clean["cast_clean"] = movies_actors_clean["cast"].str.split(r'\s*,\s*')
    actors_flat = movies_actors_clean["cast_clean"].explode()
    actor_counts = actors_flat.value_counts()
    top_actors = actor_counts.head(20)
    print("\n--- Top 20 Actors ---\n", top_actors)
    return top_actors

def get_popular_countries_data(df: pd.DataFrame) -> pd.Series:
    """Extracts data for the most popular countries."""
    movies_country_clean = df.dropna(subset="country").copy()
    print(f"\n--- Null values in 'country' after dropping ---\n {movies_country_clean['country'].isnull().sum()}")
    movies_country_clean["countries_separated"] = movies_country_clean["country"].str.split(r'\s*,\s*')
    
    countries_flat = movies_country_clean["countries_separated"].explode()
    country_counts = countries_flat.value_counts()
    top_countries = country_counts.head(20)
    print("\n--- Top 20 Countries by Content ---\n", top_countries)
    return top_countries

def get_popular_movie_genres_data(df: pd.DataFrame) -> pd.Series:
    """Extracts data for the most popular movie genres."""
    print(f"\n--- Null values in 'listed_in' (genres) for movies ---\n {df['listed_in'].isnull().sum()}")
    movies_genre_clean = df.dropna(subset="listed_in").copy()
    movies_genre_clean_movies = movies_genre_clean[movies_genre_clean["type"] == "Movie"].copy()
    movies_genre_clean_movies["genres"] = movies_genre_clean_movies["listed_in"].str.split(r'\s*,\s*')
    genre_flat = movies_genre_clean_movies["genres"].explode()
    genre_counts = genre_flat.value_counts()
    top_movie_genres = genre_counts.head(20)
    print("\n--- Top 20 Movie Genres ---\n", top_movie_genres)
    return top_movie_genres

def get_popular_tvshow_genres_data(df: pd.DataFrame) -> pd.Series:
    """Extracts data for the most popular TV show genres."""
    tvshows_genre_clean = df.dropna(subset="listed_in").copy()
    tvshows_genre_clean_shows = tvshows_genre_clean[tvshows_genre_clean["type"] == "TV Show"].copy()
    tvshows_genre_clean_shows["genres"] = tvshows_genre_clean_shows["listed_in"].str.split(r'\s*,\s*')
    genre_flat = tvshows_genre_clean_shows["genres"].explode()
    genre_counts = genre_flat.value_counts()
    top_tvshows_genres = genre_counts.head(20)
    print("\n--- Top 20 TV Show Genres ---\n", top_tvshows_genres)
    return top_tvshows_genres

def get_ratings_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts data for ratings distribution."""
    print(f"\n--- Null values in 'rating' ---\n {df['rating'].isnull().sum()}")
    rating_clean = df.dropna(subset="rating").copy()
    rating_movies_shows = rating_clean.groupby(["rating", "type"]).size().unstack(level=-1, fill_value=0)
    print("\n--- Ratings Distribution for Movies and TV Shows ---\n", rating_movies_shows)
    return rating_movies_shows


# --- Plotting Functions ---

def plot_content_type_distribution(type_counts: pd.Series):
    """Plots the distribution of Movies and TV Shows using pie and bar charts."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.pie(type_counts, labels=type_counts.index, autopct="%1.1f%%", startangle=90, colors=["skyblue", "lightcoral"])
    plt.title("Distribution of Movies and TV Shows")
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    sns.barplot(x=type_counts.index, y=type_counts.values, palette=["skyblue", "lightcoral"])
    plt.title("Distribution of Movies and TV Shows")
    plt.xlabel("Type of Show")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_yearly_cumulative_content(type_added_by_year: pd.DataFrame):
    """Plots the cumulative number of Movies and TV Shows added over years."""
    cumulative_counts = type_added_by_year.cumsum()
    print("\n--- Cumulative Content Added by Year ---\n", cumulative_counts) 

    cumulative_counts.plot(kind="line", marker="o", figsize=(10, 6), color=["skyblue", "pink"])
    plt.title("Cumulative Number of Movies and TV Shows Added Over Years")
    plt.xlabel("Year Added")
    plt.ylabel("Cumulative Count")
    plt.grid(True)
    plt.legend(title="Content Type")
    plt.tight_layout()
    plt.show()

def plot_yearly_content_bars(type_added_by_year: pd.DataFrame):
    """Plots the annual number of Movies and TV Shows added per year."""
    type_added_by_year.plot(kind="bar", alpha=0.7, figsize=(10, 6), color=["skyblue", "pink"])
    plt.xlabel("Year Added")
    plt.ylabel("Count")
    plt.grid(True, linewidth=0.5)
    plt.title("Number of TV Shows and Movies Added to Netflix per Year")
    plt.xticks(rotation=45)
    plt.legend(title="Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_monthly_content_bars(type_added_by_month: pd.DataFrame):
    """Plots the monthly number of Movies and TV Shows added."""
    type_added_by_month.plot(kind="bar", alpha=0.7, figsize=(10, 6), color=["skyblue", "pink"])
    plt.xlabel("Month Added")
    plt.ylabel("Count")
    plt.grid(True, linewidth=0.5)
    plt.title("Number of TV Shows and Movies Added to Netflix per Month")
    plt.xticks(rotation=45)
    plt.legend(title="Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_movie_duration_distribution(movies_duration_df: pd.DataFrame):
    """Plots the distribution of movie durations."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=movies_duration_df, x="duration_minutes", bins=50, kde=True, color="mediumseagreen")
    plt.title("Distribution of Movie Durations on Netflix")
    plt.xlabel("Duration (minutes)")
    mean_duration = movies_duration_df["duration_minutes"].mean()
    plt.axvline(mean_duration, linestyle="--", color="mediumorchid", label=f"Mean: {mean_duration:.1f} min")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_tv_show_season_distribution(shows_by_seasons: pd.Series):
    """Plots the distribution of TV shows by number of seasons."""
    plt.figure(figsize=(10, 6))
    shows_by_seasons.plot(kind="bar", color="mediumslateblue", alpha=0.8)
    plt.xlabel("Number of Seasons")
    plt.ylabel("Number of TV Shows")
    plt.title("Distribution of TV Shows by Number of Seasons")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    for index, value in enumerate(shows_by_seasons):
        plt.text(index, value + 1, str(value), ha="center", va="bottom", fontsize=10)
    plt.show()

def plot_most_popular_directors(most_popular_directors: pd.Series):
    """Plots the most popular directors."""
    plt.figure(figsize=(10, 6))
    most_popular_directors.plot(kind="bar", color="mediumslateblue")
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.title("10 Most Popular Directors on Netflix")
    plt.xlabel("Director")
    plt.ylabel("Number of Titles")
    for index, value in enumerate(most_popular_directors):
        plt.text(index, value, str(value), ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

def plot_most_popular_actors(top_actors: pd.Series):
    """Plots the most popular actors."""
    plt.figure(figsize=(12, 6))
    top_actors.plot(kind="bar", color="skyblue")
    plt.title("Top 20 Actors on Netflix")
    plt.xlabel("Actor")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.xticks(rotation=60, ha="right")
    plt.show()

def plot_most_popular_countries(top_countries: pd.Series):
    """Plots the most popular countries by content frequency."""
    plt.figure(figsize=(10, 6))
    top_countries.plot(kind="bar", color='skyblue')
    plt.title("Top 20 Countries by Content Frequency on Netflix")
    plt.xlabel("Country")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    for index, value in enumerate(top_countries):
        plt.text(index, value, str(value), ha="center", va="bottom")
    plt.show()

def plot_most_popular_genres_movies(top_movie_genres: pd.Series):
    """Plots the most popular genres for movies."""
    plt.figure(figsize=(12, 6))
    top_movie_genres.plot(kind="bar", color="skyblue")
    plt.title("Top 20 Movie Genres on Netflix")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.xticks(rotation=60, ha="right")
    plt.show()

def plot_most_popular_genres_tvshows(top_tvshows_genres: pd.Series):
    """Plots the most popular genres for TV shows."""
    plt.figure(figsize=(12, 6))
    top_tvshows_genres.plot(kind="bar", color="skyblue")
    plt.title("Top 20 TV Show Genres on Netflix")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.xticks(rotation=60, ha="right")
    plt.show()

def plot_ratings_distribution(rating_movies_shows: pd.DataFrame):
    """Plots the distribution of ratings for movies and TV shows."""
    plt.figure(figsize=(10, 6))
    rating_movies_shows.plot(kind="bar", ax=plt.gca(), color=["purple", "green"])
    plt.grid(True, linewidth=0.5)
    plt.title("Ratings for Movies and TV Shows on Netflix")
    plt.xlabel("Ratings")
    plt.ylabel("Frequency")
    plt.legend(["TV Shows", "Movies"])
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    DATA_PATH = r"./Netflix_movies_and_tv_shows_clustering.csv"

    # 1. Data Loading and Initial Overview
    netflix_df = load_data(DATA_PATH)
    get_initial_overview(netflix_df)

    # 2. Data Cleaning and Feature Engineering
    netflix_df = clean_and_extract_dates(netflix_df)
    netflix_df = clean_duration_and_seasons(netflix_df)
    get_null_value_summary(netflix_df) 

    # 3. Prepare Data for Visualizations (Analysis)
    content_type_data = get_content_type_data(netflix_df)
    yearly_content_data = get_yearly_content_data(netflix_df)
    monthly_content_data = get_monthly_content_data(netflix_df)
    movie_duration_data = get_movie_duration_data(netflix_df)
    tv_show_season_data = get_tv_show_season_data(netflix_df)
    get_oldest_content_data(netflix_df)
    popular_directors_data = get_popular_directors_data(netflix_df)
    popular_actors_data = get_popular_actors_data(netflix_df)
    popular_countries_data = get_popular_countries_data(netflix_df)
    popular_movie_genres_data = get_popular_movie_genres_data(netflix_df)
    popular_tvshow_genres_data = get_popular_tvshow_genres_data(netflix_df)
    ratings_data = get_ratings_data(netflix_df)

    # 4. Generate Visualizations
    plot_content_type_distribution(content_type_data)
    plot_yearly_cumulative_content(yearly_content_data)
    plot_yearly_content_bars(yearly_content_data)
    plot_monthly_content_bars(monthly_content_data)
    plot_movie_duration_distribution(movie_duration_data)
    plot_tv_show_season_distribution(tv_show_season_data)
    plot_most_popular_directors(popular_directors_data)
    plot_most_popular_actors(popular_actors_data)
    plot_most_popular_countries(popular_countries_data)
    plot_most_popular_genres_movies(popular_movie_genres_data)
    plot_most_popular_genres_tvshows(popular_tvshow_genres_data)
    plot_ratings_distribution(ratings_data)
