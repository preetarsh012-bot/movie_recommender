  import pandas as pd
import ast
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------
# 1. LOAD DATASETS
# ----------------------------------

movies = pd.read_csv(
    "../data/tmdb_5000_movies.csv"
)

credits = pd.read_csv(
    "../data/tmdb_5000_credits.csv"
)


# ----------------------------------
# 2. MERGE DATASETS
# ----------------------------------

movies = movies.merge(
    credits,
    on="title"
)


# ----------------------------------
# 3. SELECT IMPORTANT COLUMNS
# ----------------------------------

movies = movies[
    [
        "movie_id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew"
    ]
]


# ----------------------------------
# 4. REMOVE MISSING VALUES
# ----------------------------------

movies.dropna(
    inplace=True
)


# ----------------------------------
# 5. CONVERT GENRES AND KEYWORDS
# ----------------------------------

def convert(text):

    result = []

    for item in ast.literal_eval(text):
        result.append(item["name"])

    return result


movies["genres"] = movies[
    "genres"
].apply(convert)

movies["keywords"] = movies[
    "keywords"
].apply(convert)


# ----------------------------------
# 6. GET TOP 3 CAST MEMBERS
# ----------------------------------

def get_cast(text):

    result = []
    counter = 0

    for item in ast.literal_eval(text):

        if counter < 3:
            result.append(
                item["name"]
            )
            counter += 1

    return result


movies["cast"] = movies[
    "cast"
].apply(get_cast)


# ----------------------------------
# 7. GET DIRECTOR
# ----------------------------------

def get_director(text):

    result = []

    for item in ast.literal_eval(text):

        if item["job"] == "Director":

            result.append(
                item["name"]
            )

            break

    return result


movies["crew"] = movies[
    "crew"
].apply(get_director)


# ----------------------------------
# 8. CONVERT OVERVIEW INTO WORD LIST
# ----------------------------------

movies["overview"] = movies[
    "overview"
].apply(
    lambda x: x.split()
)


# ----------------------------------
# 9. REMOVE SPACES FROM NAMES
# ----------------------------------

movies["genres"] = movies[
    "genres"
].apply(
    lambda x: [
        word.replace(" ", "")
        for word in x
    ]
)

movies["keywords"] = movies[
    "keywords"
].apply(
    lambda x: [
        word.replace(" ", "")
        for word in x
    ]
)

movies["cast"] = movies[
    "cast"
].apply(
    lambda x: [
        word.replace(" ", "")
        for word in x
    ]
)

movies["crew"] = movies[
    "crew"
].apply(
    lambda x: [
        word.replace(" ", "")
        for word in x
    ]
)


# ----------------------------------
# 10. CREATE TAGS
# ----------------------------------

movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)


# ----------------------------------
# 11. CREATE FINAL DATAFRAME
# ----------------------------------

new_movies = movies[
    [
        "movie_id",
        "title",
        "tags"
    ]
]


# ----------------------------------
# 12. CONVERT LIST INTO TEXT
# ----------------------------------

new_movies = new_movies.copy()

new_movies["tags"] = new_movies[
    "tags"
].apply(
    lambda x: " ".join(x)
)


# ----------------------------------
# 13. CONVERT TEXT INTO VECTORS
# ----------------------------------

cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = cv.fit_transform(
    new_movies["tags"]
).toarray()


# ----------------------------------
# 14. CALCULATE SIMILARITY
# ----------------------------------

similarity = cosine_similarity(
    vectors
)


# ----------------------------------
# 15. SAVE MODEL
# ----------------------------------

pickle.dump(
    new_movies,
    open(
        "../models/movies.pkl",
        "wb"
    )
)

pickle.dump(
    similarity,
    open(
        "../models/similarity.pkl",
        "wb"
    )
)


print(
    "Movie recommendation model created successfully!"
)

print(
    "Total movies:",
    new_movies.shape[0]
)