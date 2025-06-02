import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

movies = pd.read_csv(r"C:\Users\alber\Downloads\Netflix_movies_and_tv_shows_clustering.csv\Netflix_movies_and_tv_shows_clustering.csv")

#Overview of the dataset
print(movies.head())
print(movies.tail())
print(list(movies.columns))
print(movies.info())

#DATA CLEANING: adjust the date_added column and create two others more, year and month
movies["date_added"] = movies["date_added"].str.strip()
movies["date_added"] = pd.to_datetime(movies["date_added"])
movies["year_added"] = movies["date_added"].dt.year   # year into year_added
movies["month_added"] = movies["date_added"].dt.month  # month into month_added
print(movies[["date_added", "year_added", "month_added"]].head())

#DATA CLEANING: adjust the duration column for TV Series and Movies
movies["duration_minutes"] = movies["duration"].apply(lambda x: int(x.split(" ")[0]) if "min" in x else None)
movies["seasons"] = movies["duration"].apply(lambda x: int(x.split(" ")[0]) if "min" not in x else None)
movies = movies.drop(columns=['duration'])
print(movies.head())
print(movies.tail())

#DATA CLEANING: relevant presence of null values
print(f"Null values for each variable:\n {movies.isnull().sum()}")

movie_variables = list(movies.columns)
for column in movie_variables:
    null_values = ((movies[column].isnull().sum()/movies.shape[0]) * 100).round(2)
    print(f"Share of null values for variable {column} on total sample: {null_values}")


#Analysis of the Type of observation (Movie - TV Serie) through graphs
print(f"Null values in types column:\n {movies["type"].isnull().sum()}")
type_counts = movies.groupby("type").size()
print(type_counts)
type_share = (type_counts/(movies.shape[0])*100).round(2)
print(type_share)
print(list(type_counts.index))

plt.pie(type_counts, labels = type_counts.index, autopct="%1.1f%%", startangle=90, colors=["skyblue", "lightcoral"])
plt.title("Distribution of Movies and TV Shows")
plt.axis("equal")
plt.show()

sns.countplot(data = movies, x = "type", color= "skyblue")
plt.title("Distribution of Movies and TV Shows")
plt.xlabel("Type of Show")
plt.ylabel("Frequency")
plt.show()


#Trend over the years of movies and tv series added on the platform (cumulative)
movies = movies.sort_values(by=["year_added", "month_added"], ascending=True)
print(movies.head())
print(movies[["year_added", "month_added"]].tail())

movies_clean = movies.dropna(subset=["year_added", "month_added"])
movies_clean["year_added"] = movies_clean["year_added"].astype(int) #TO AVOID 2010.0
movies_clean["month_added"] = movies_clean["month_added"].astype(int) #TO AVOID 11.0
print(f"Null values: {movies_clean[["year_added", "month_added"]].isnull().sum()}")
type_added_by_year = movies_clean.groupby(["year_added", "type"]).size().unstack(fill_value=0)
print(type_added_by_year)

cumulative_counts = type_added_by_year.cumsum()
print(cumulative_counts)

cumulative_counts.plot(kind="line", marker="o", figsize=(10, 6), color = ["skyblue", "pink"]) #use .plot directly on the dataframe, cumulative_counts is a dataframe
plt.title("Cumulative Number of Movies and TV Shows Added Over Years")
plt.xlabel("Year Added")
plt.ylabel("Cumulative Count")
plt.grid(True)
plt.legend(title="Content Type")
plt.tight_layout() # it adjusts the spacing between subplots in a figure so that labels, titles, and tick labels don't overlap or get cut off
plt.show()

#Year and month (separately) with the most movies and tv series added
type_added_by_year.plot(kind="bar", alpha=0.7, figsize=(10,6), color = ["skyblue", "pink"])
plt.xlabel("Year Added")
plt.ylabel("Count")
plt.grid(True, linewidth=0.5)
plt.title("Number of TV Shows and Movies Added to Netflix per Year")
plt.xticks(rotation=45)
plt.legend(title="Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

#

type_added_by_month = movies_clean.groupby(["month_added", "type"]).size().unstack(fill_value=0)
print(type_added_by_month)
type_added_by_month.plot(kind="bar", alpha=0.7, figsize=(10,6), color = ["skyblue", "pink"])
plt.xlabel("Month Added")
plt.ylabel("Count")
plt.grid(True, linewidth=0.5)
plt.title("Number of TV Shows and Movies Added to Netflix per Month")
plt.xticks(rotation=45)
plt.legend(title="Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


#Duration and seasons, distribution for movies and tv series
print(movies[movies["type"] == "Movie"]["duration_minutes"].isnull().sum()) #right order, to look for null values 
movies_duration_df = movies[movies["type"] == "Movie"][["type", "duration_minutes"]] #filter only movies, than take only type and duration columns
print(movies_duration_df["type"].unique()) #check if filter worked
print(movies_duration_df.head())
plt.figure(figsize=(10, 6))
sns.histplot(data=movies_duration_df, x="duration_minutes", bins=50, kde=True, color="mediumseagreen")
plt.title("Distribution of Movie Durations on Netflix")
plt.xlabel("Duration (minutes)")
mean_duration = movies_duration_df["duration_minutes"].mean()
plt.axvline(mean_duration, linestyle="--", color="mediumorchid", label=f"Mean: {mean_duration:.1f} min")
plt.legend()
plt.tight_layout()
plt.show()

shows_seasons_df = movies[movies["type"] == "TV Show"][["type", "seasons"]]
shows_seasons_df["seasons"] = shows_seasons_df["seasons"].astype(int)
print(shows_seasons_df.head())
shows_seasons_df["season_range"]=shows_seasons_df["seasons"].apply(
    lambda x: "1-2 seasons" if x < 3 else
                "3-5 seasons" if x > 2 and x < 6 else
                "6-10 seasons" if x > 5 and x < 11 else
                "11+ seasons"
)
print(shows_seasons_df.head())
season_order = ["1-2 seasons", "3-5 seasons", "6-10 seasons", "11+ seasons"]
shows_seasons_df["season_range"] = pd.Categorical(
    shows_seasons_df["season_range"],
    categories=season_order,
    ordered=True
)
shows_by_seasons = shows_seasons_df.groupby("season_range").size()
print(shows_by_seasons)
shows_by_seasons.plot(kind="bar", color ="mediumslateblue", alpha=0.8, figsize=(10, 6))
plt.xlabel("Number of Seasons")
plt.ylabel("Number of TV Shows")
plt.title("Distribution of TV Shows by Number of Seasons")
plt.grid(axis="y", linestyle="--", linewidth=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
for index, value in enumerate(shows_by_seasons):
    plt.text(index, value + 1, str(value), ha="center", va="bottom", fontsize=10)
plt.show()

#Release year: oldest tv shows and movies
movies_df = movies[movies["type"] == "Movie"]
oldest_movies = movies_df.sort_values(["release_year", "title"])
print(f"Oldest Movies on Netflix:\n {oldest_movies[["show_id", "title", "type","release_year"]][:15]}")

tvshows_df = movies[movies["type"] == "TV Show"]
oldest_tvshows = tvshows_df.sort_values(["release_year", "title"])
print(f"Oldest TV Shows on Netflix:\n {oldest_tvshows[["show_id", "title", "type","release_year"]][:15]}")

#Most popular directors on Netflix
movies_director_clean = movies.dropna(subset = "director")
print(movies_director_clean["director"].isnull().sum())

most_popular_directors = movies_director_clean["director"].value_counts()[:10]
print(f"10 Most popular directors on Netflix (2019):\n {most_popular_directors}")
most_popular_directors = most_popular_directors.astype(int) #did not work

most_popular_directors.plot(kind = "bar", color ="mediumslateblue")
import matplotlib.ticker as ticker
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.title("10 most popular directors on Netflix (2019)")
plt.xlabel("Director")
plt.ylabel("No. of Movies")
for index, value in enumerate(most_popular_directors):
    plt.text(index, value, str(value), ha = "center", va = "bottom", fontsize = 8)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

#Most popular actors on Netflix
print(f"Null values in cast: \n {movies["cast"].isnull().sum()}")
movies_actors_clean = movies.dropna(subset = "cast")
print(f"Null values in cast: \n {movies_actors_clean["cast"].isnull().sum()}")

movies_actors_clean["cast_clean"] = movies_actors_clean["cast"].str.split(r'\s*,\s*')
print(movies_actors_clean[["cast", "cast_clean"]][:20])

actors_flat = movies_actors_clean["cast_clean"].explode()
actor_counts = actors_flat.value_counts()
top_actors = actor_counts[:20]

plt.figure(figsize=(12, 6)) 
top_actors.plot(kind="bar", color = "skyblue")
plt.title("Top 20 Actors")
plt.xlabel("Actor")
plt.ylabel("Count")
plt.tight_layout()
plt.xticks(rotation=60, ha = "right")
plt.show()

#Most popular countries
movies_country_clean = movies.dropna(subset = "country")
print(movies_country_clean["country"].isnull().sum())
movies_country_clean["countries_separated"] = movies_country_clean["country"].str.split(r'\s*,\s*')
print(movies_country_clean[:15])

countries_flat = movies_country_clean["countries_separated"].explode()
print(countries_flat)
country_counts = countries_flat.value_counts()
top_countries = country_counts[:20]
print(top_countries)

plt.figure(figsize=(10, 6))
top_countries.plot(kind="bar", color='skyblue')
plt.title("Top 20 Countries by Frequency in Movies")
plt.xlabel("Country")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
for index, value in enumerate(top_countries):
    plt.text(index, value, str(value), ha = "center", va = "bottom")
plt.show()


#Most Popular genres for Movies
print(f"Null values in genre: \n {movies["listed_in"].isnull().sum()}")
movies_genre_clean = movies.dropna(subset = "listed_in")
print(f"Null values in genre: \n {movies_genre_clean["listed_in"].isnull().sum()}")

movies_genre_clean_movies = movies_genre_clean[movies_genre_clean["type"] == "Movie"]
print(movies_genre_clean_movies["type"].unique())
print(list(movies_genre_clean_movies.columns))
movies_genre_clean_movies["genres"] = movies_genre_clean_movies["listed_in"].str.split(r'\s*,\s*')
print(movies_genre_clean_movies[["listed_in", "genres"]][:15])
genre_flat = movies_genre_clean_movies["genres"].explode()
genre_counts = genre_flat.value_counts()
top_movie_genres = genre_counts[:20]

plt.figure(figsize=(12, 6)) 
top_movie_genres.plot(kind = "bar", color = "skyblue")
plt.title("Top 20 Movie genres")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.tight_layout()
plt.xticks(rotation=60, ha = "right")
plt.show()

#Most Popular genres for TV Shows

tvshows_genre_clean_movies = movies_genre_clean[movies_genre_clean["type"] == "TV Show"]
print(tvshows_genre_clean_movies["type"].unique())
tvshows_genre_clean_movies["genres"] = tvshows_genre_clean_movies["listed_in"].str.split(r'\s*,\s*')
print(tvshows_genre_clean_movies[["listed_in", "genres"]][:15])
genre1_flat = tvshows_genre_clean_movies["genres"].explode()
genre1_counts = genre1_flat.value_counts()
top_tvshows_genres = genre1_counts[:20]

plt.figure(figsize=(12, 6)) 
top_tvshows_genres.plot(kind = "bar", color = "skyblue")
plt.title("Top 20 TV Shows genres")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.tight_layout()
plt.xticks(rotation=60, ha = "right")
plt.show()

#Ratings for movies and tv shows
print(movies["type"].isnull().sum())
print(movies["rating"].isnull().sum())
rating_clean = movies.dropna(subset = "rating")

rating_movies_shows = rating_clean.groupby(["rating", "type"]).size().unstack(level = -1, fill_value=0)
print(rating_movies_shows)

rating_movies_shows.plot(kind="bar", figsize = (10, 6), color = ["purple", "green"])
plt.grid(True, linewidth = 0.5)
plt.title("Ratings for Movies and TV Shows on Netflix")
plt.xlabel("Ratings")
plt.ylabel("Frequency")
plt.legend(["TV Shows", "Movies"])
plt.tight_layout()
plt.xticks(rotation = 45)
plt.show()












