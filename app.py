from flask import Flask, render_template, request, redirect, url_for, session as flask_session
import pickle
import requests
import os
import sqlite3

from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from urllib.parse import quote_plus


# ============================================================
# PROJECT BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(
    os.path.join(BASE_DIR, ".env")
)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE_URL = "https://api.themoviedb.org/3"

TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "movie-recommender-local-secret"
)


# ============================================================
# DATABASE
# ============================================================

DATABASE_PATH = os.path.join(BASE_DIR, "movie_recommender.db")


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


tmdb_session = requests.Session()


# ============================================================
# CURRENT USER
# ============================================================

def current_user_id():

    return flask_session.get(
        "user_id"
    )


# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(route_function):

    @wraps(route_function)
    def wrapped_route(*args, **kwargs):

        if "user_id" not in flask_session:

            return redirect(
                url_for(
                    "login",
                    next=request.path
                )
            )

        return route_function(
            *args,
            **kwargs
        )

    return wrapped_route


# ============================================================
# CREATE DATABASE TABLES
def create_tables():
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_name TEXT NOT NULL,
            movie_id INTEGER,
            searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_name TEXT NOT NULL,
            movie_id INTEGER,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (user_id, movie_name),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    connection.commit()
    connection.close()


create_tables()


# LOAD MACHINE LEARNING MODELS
# ============================================================

with open(
    os.path.join(
        BASE_DIR,
        "models",
        "movies.pkl"
    ),
    "rb"
) as file:

    movies = pickle.load(file)


SIMILARITY_URL = (
    "https://huggingface.co/coderancer/movie-recommender-model/resolve/main/similarity.pkl"
)


LOCAL_SIMILARITY_FILE = os.path.join(
    BASE_DIR,
    "models",
    "similarity.pkl"
)


TEMP_SIMILARITY_FILE = os.path.join(
    os.path.expanduser("~"),
    "movie_recommender_similarity.pkl"
)


# ============================================================
# LOAD SIMILARITY MODEL
# ============================================================

if (
    os.path.exists(LOCAL_SIMILARITY_FILE)
    and
    os.path.getsize(LOCAL_SIMILARITY_FILE)
    > 10 * 1024 * 1024
):

    SIMILARITY_FILE = LOCAL_SIMILARITY_FILE

else:

    SIMILARITY_FILE = TEMP_SIMILARITY_FILE

    if not os.path.exists(SIMILARITY_FILE):

        print(
            "Downloading similarity model..."
        )

        response = requests.get(
            SIMILARITY_URL,
            stream=True,
            timeout=300
        )

        response.raise_for_status()

        with open(
            SIMILARITY_FILE,
            "wb"
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    file.write(chunk)


# ============================================================
# LOAD SIMILARITY
# ============================================================

with open(
    SIMILARITY_FILE,
    "rb"
) as file:

    similarity = pickle.load(file)


# ============================================================
# NORMALIZE TITLE
# ============================================================

def normalize_title(title):

    if not title:

        return ""

    return (
        str(title)
        .strip()
        .lower()
    )


# ============================================================
# FIND MOVIE INDEX
# ============================================================

def find_movie_index(movie_name):

    search_name = normalize_title(
        movie_name
    )

    for index, title in enumerate(
        movies["title"]
    ):

        if normalize_title(
            title
        ) == search_name:

            return index

    return None


# ============================================================
# TMDB REQUEST
# ============================================================

def tmdb_request(
    endpoint,
    params=None
):

    if not TMDB_API_KEY:

        print(
            "TMDB API key not found."
        )

        return None


    if params is None:

        params = {}


    params["api_key"] = TMDB_API_KEY


    url = (
        f"{TMDB_BASE_URL}/{endpoint}"
    )


    try:

        response = tmdb_session.get(
            url,
            params=params,
            timeout=15
        )


        if response.status_code != 200:

            print(
                "TMDB Error:",
                response.status_code,
                response.text[:300]
            )

            return None


        return response.json()


    except requests.exceptions.RequestException as error:

        print(
            "TMDB connection error:",
            error
        )

        return None


# ============================================================
# SEARCH MOVIE ON TMDB
# ============================================================

def search_tmdb_movie(movie_name):

    data = tmdb_request(
        "search/movie",
        {
            "query": movie_name,
            "language": "en-US",
            "page": 1,
            "include_adult": False
        }
    )


    if not data:

        return None


    results = data.get(
        "results",
        []
    )


    if not results:

        return None


    search_name = normalize_title(
        movie_name
    )


    for movie in results:

        if normalize_title(
            movie.get("title")
        ) == search_name:

            return movie


    return results[0]


# ============================================================
# GET COMPLETE TMDB MOVIE DETAILS
# ============================================================

def get_tmdb_movie_details(movie_id):

    data = tmdb_request(
        f"movie/{movie_id}",
        {
            "language": "en-US"
        }
    )


    if not data:

        return None


    poster_path = data.get(
        "poster_path"
    )


    if poster_path:

        poster_url = (
            TMDB_IMAGE_URL
            + poster_path
        )

    else:

        poster_url = None


    genres = ", ".join(
        genre.get(
            "name",
            ""
        )
        for genre in data.get(
            "genres",
            []
        )
    )


    trailer = get_trailer_link(
        movie_id,
        data.get(
            "title",
            ""
        )
    )


    return {

        "movie_id": data.get(
            "id"
        ),

        "title": data.get(
            "title",
            "Unknown"
        ),

        "poster": poster_url,

        "overview": data.get(
            "overview",
            "Overview not available."
        ),

        "rating": round(
            float(
                data.get(
                    "vote_average",
                    0
                )
            ),
            1
        ),

        "release_date": data.get(
            "release_date",
            "Not available"
        ),

        "genres": genres,

        "trailer": trailer

    }


# ============================================================
# GET MOVIE DETAILS BY TITLE
# ============================================================

def get_movie_details(movie_name):

    tmdb_movie = search_tmdb_movie(
        movie_name
    )


    if not tmdb_movie:

        return create_empty_movie(
            movie_name
        )


    movie_id = tmdb_movie.get(
        "id"
    )


    details = get_tmdb_movie_details(
        movie_id
    )


    if not details:

        return create_empty_movie(
            movie_name
        )


    return details


# ============================================================
# YOUTUBE SEARCH LINK
# ============================================================

def youtube_search_link(movie_name):

    query = quote_plus(
        movie_name
        + " official trailer"
    )


    return (
        "https://www.youtube.com/results"
        "?search_query="
        + query
    )


# ============================================================
# GET TRAILER LINK
# ============================================================

def get_trailer_link(
    movie_id,
    movie_name
):

    backup_link = youtube_search_link(
        movie_name
    )


    data = tmdb_request(
        f"movie/{movie_id}/videos",
        {
            "language": "en-US"
        }
    )


    if not data:

        return backup_link


    videos = data.get(
        "results",
        []
    )


    # --------------------------------------------------------
    # OFFICIAL YOUTUBE TRAILER
    # --------------------------------------------------------

    for video in videos:

        if (
            video.get("site") == "YouTube"
            and
            video.get("type") == "Trailer"
            and
            video.get("official") is True
        ):

            key = video.get(
                "key"
            )


            if key:

                return (
                    "https://www.youtube.com/watch?v="
                    + key
                )


    # --------------------------------------------------------
    # ANY YOUTUBE TRAILER
    # --------------------------------------------------------

    for video in videos:

        if (
            video.get("site") == "YouTube"
            and
            video.get("type") == "Trailer"
        ):

            key = video.get(
                "key"
            )


            if key:

                return (
                    "https://www.youtube.com/watch?v="
                    + key
                )


    return backup_link


# ============================================================
# EMPTY MOVIE DETAILS
# ============================================================

def create_empty_movie(movie_name):

    return {

        "movie_id": None,

        "title": movie_name,

        "poster": None,

        "overview":
            "Movie details could not be loaded.",

        "rating": "N/A",

        "release_date": "N/A",

        "genres": "N/A",

        "trailer":
            youtube_search_link(
                movie_name
            )

    }


# ============================================================
# FAVORITE CHECK
# ============================================================

def is_favorite(movie_name):

    user_id = current_user_id()


    if user_id is None:

        return False


    connection = get_db_connection()


    try:

        result = connection.execute(
            """
            SELECT id
            FROM favorites
            WHERE user_id = ?
            AND LOWER(movie_name) = LOWER(?)
            """,
            (
                user_id,
                movie_name
            )
        ).fetchone()

    finally:

        connection.close()


    return result is not None


# ============================================================
# ADD SEARCH HISTORY
# ============================================================

def add_search_history(movie_name):

    user_id = current_user_id()


    if user_id is None:

        return


    connection = get_db_connection()


    try:

        connection.execute(
            """
            INSERT INTO search_history
            (
                movie_name,
                user_id
            )
            VALUES (?, ?)
            """,
            (
                movie_name,
                user_id
            )
        )


        connection.commit()

    except Exception as error:

        connection.rollback()

        print(
            "History error:",
            error
        )

    finally:

        connection.close()


# ============================================================
# GET ML RECOMMENDATIONS
# ============================================================

def get_ml_recommendations(
    movie_name,
    limit=5
):

    movie_index = find_movie_index(
        movie_name
    )


    if movie_index is None:

        return []


    try:

        distances = similarity[
            movie_index
        ]


        movie_list = sorted(
            list(
                enumerate(distances)
            ),
            reverse=True,
            key=lambda x: x[1]
        )


    except Exception as error:

        print(
            "ML recommendation error:",
            error
        )

        return []


    recommendations = []


    for index, score in movie_list:

        if index == movie_index:

            continue


        try:

            movie_title = str(
                movies.iloc[index]["title"]
            )

        except Exception:

            continue


        details = get_movie_details(
            movie_title
        )


        if not details:

            continue


        try:

            numeric_score = float(
                score
            )


            if 0 <= numeric_score <= 1:

                numeric_score *= 100


            numeric_score = round(
                numeric_score,
                1
            )

        except Exception:

            numeric_score = None


        details["title"] = movie_title

        details["similarity"] = numeric_score

        details["recommendation_source"] = "AI"

        details["favorite"] = is_favorite(
            movie_title
        )


        recommendations.append(
            details
        )


        if len(recommendations) >= limit:

            break


    return recommendations


# ============================================================
# GET TMDB RECOMMENDATIONS
# ============================================================

def get_tmdb_recommendations(
    movie_id,
    limit=5
):

    data = tmdb_request(
        f"movie/{movie_id}/recommendations",
        {
            "language": "en-US",
            "page": 1
        }
    )


    if not data:

        return []


    results = data.get(
        "results",
        []
    )


    recommendations = []


    for movie in results:

        if len(recommendations) >= limit:

            break


        tmdb_id = movie.get(
            "id"
        )


        title = movie.get(
            "title"
        )


        if not tmdb_id or not title:

            continue


        details = get_tmdb_movie_details(
            tmdb_id
        )


        if not details:

            continue


        details["recommendation_source"] = "TMDB"

        details["similarity"] = None

        details["favorite"] = is_favorite(
            title
        )


        recommendations.append(
            details
        )


    return recommendations


# ============================================================
# GET FRANCHISE / SEQUEL RECOMMENDATIONS
# ============================================================

def get_franchise_recommendations(
    movie_id
):

    movie_data = tmdb_request(
        f"movie/{movie_id}",
        {
            "language": "en-US"
        }
    )


    if not movie_data:

        return []


    collection = movie_data.get(
        "belongs_to_collection"
    )


    if not collection:

        return []


    collection_id = collection.get(
        "id"
    )


    if not collection_id:

        return []


    collection_data = tmdb_request(
        f"collection/{collection_id}",
        {
            "language": "en-US"
        }
    )


    if not collection_data:

        return []


    parts = collection_data.get(
        "parts",
        []
    )


    selected_title = normalize_title(
        movie_data.get(
            "title",
            ""
        )
    )


    selected_release_date = movie_data.get(
        "release_date",
        ""
    )


    future_sequels = []

    all_other_parts = []


    for part in parts:

        part_title = normalize_title(
            part.get(
                "title",
                ""
            )
        )


        if not part_title:

            continue


        if part_title == selected_title:

            continue


        part_release_date = part.get(
            "release_date",
            ""
        )


        if (
            selected_release_date
            and
            part_release_date
            and
            part_release_date > selected_release_date
        ):

            future_sequels.append(
                part
            )

        else:

            all_other_parts.append(
                part
            )


    future_sequels.sort(
        key=lambda movie:
        movie.get(
            "release_date",
            ""
        )
    )


    if not future_sequels:

        all_other_parts.sort(
            key=lambda movie:
            movie.get(
                "release_date",
                ""
            )
        )

        selected_parts = all_other_parts

    else:

        selected_parts = future_sequels


    recommendations = []


    for part in selected_parts:

        if len(recommendations) >= 5:

            break


        part_id = part.get(
            "id"
        )


        if not part_id:

            continue


        details = get_tmdb_movie_details(
            part_id
        )


        if not details:

            continue


        details["recommendation_source"] = (
            "Sequel"
        )

        details["similarity"] = None

        details["favorite"] = is_favorite(
            details["title"]
        )


        recommendations.append(
            details
        )


    return recommendations


# ============================================================
# REMOVE DUPLICATE MOVIES
# ============================================================

def remove_duplicate_movies(
    movie_list
):

    unique_movies = []

    seen_titles = set()


    for movie in movie_list:

        title = normalize_title(
            movie.get(
                "title",
                ""
            )
        )


        if not title:

            continue


        if title in seen_titles:

            continue


        seen_titles.add(
            title
        )


        unique_movies.append(
            movie
        )


    return unique_movies


# ============================================================
# FINAL HYBRID RECOMMENDATIONS
# ============================================================

def recommend(movie_name):

    tmdb_movie = search_tmdb_movie(
        movie_name
    )


    if not tmdb_movie:

        return []


    movie_id = tmdb_movie.get(
        "id"
    )


    if not movie_id:

        return []


    # --------------------------------------------------------
    # 1. FRANCHISE / SEQUELS
    # --------------------------------------------------------

    franchise_recommendations = (
        get_franchise_recommendations(
            movie_id
        )
    )


    # --------------------------------------------------------
    # 2. AI / ML
    # --------------------------------------------------------

    ml_recommendations = (
        get_ml_recommendations(
            movie_name,
            limit=5
        )
    )


    # --------------------------------------------------------
    # 3. TMDB
    # --------------------------------------------------------

    tmdb_recommendations = (
        get_tmdb_recommendations(
            movie_id,
            limit=5
        )
    )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    general_recommendations = (
        ml_recommendations
        +
        tmdb_recommendations
    )


    general_recommendations = (
        remove_duplicate_movies(
            general_recommendations
        )
    )


    # --------------------------------------------------------
    # REMOVE FRANCHISE DUPLICATES
    # --------------------------------------------------------

    franchise_titles = {

        normalize_title(
            movie.get(
                "title",
                ""
            )
        )

        for movie
        in franchise_recommendations

    }


    general_recommendations = [

        movie

        for movie
        in general_recommendations

        if normalize_title(
            movie.get(
                "title",
                ""
            )
        )
        not in franchise_titles

    ]


    # --------------------------------------------------------
    # FINAL MAXIMUM 5
    # --------------------------------------------------------

    final_recommendations = []


    for movie in franchise_recommendations:

        if len(final_recommendations) >= 5:

            break


        final_recommendations.append(
            movie
        )


    for movie in general_recommendations:

        if len(final_recommendations) >= 5:

            break


        final_recommendations.append(
            movie
        )


    return final_recommendations


# ============================================================
# GET HOME CATEGORY MOVIES
# ============================================================

def get_tmdb_movies(
    endpoint,
    extra_params=None
):

    params = {

        "language": "en-US",

        "page": 1

    }


    if extra_params:

        params.update(
            extra_params
        )


    data = tmdb_request(
        endpoint,
        params
    )


    if not data:

        return []


    results = data.get(
        "results",
        []
    )


    results = results[:10]


    formatted_movies = []


    for movie in results:

        movie_id = movie.get(
            "id"
        )


        title = movie.get(
            "title"
        )


        if not movie_id or not title:

            continue


        poster_path = movie.get(
            "poster_path"
        )


        if poster_path:

            poster = (
                TMDB_IMAGE_URL
                +
                poster_path
            )

        else:

            poster = None


        formatted_movies.append({

            "id": movie_id,

            "title": title,

            "poster": poster,

            "rating": round(
                float(
                    movie.get(
                        "vote_average",
                        0
                    )
                ),
                1
            ),

            "release_date":
                movie.get(
                    "release_date",
                    "N/A"
                )

        })


    return formatted_movies


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if "user_id" in flask_session:

        return redirect(
            url_for("home")
        )


    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not username or not password:

            return render_template(
                "register.html",
                error=(
                    "Username and password "
                    "are required."
                )
            )


        if len(password) < 8:

            return render_template(
                "register.html",
                error=(
                    "Password must be at least "
                    "8 characters."
                )
            )


        if password != confirm_password:

            return render_template(
                "register.html",
                error="Passwords do not match."
            )


        connection = None


        try:

            connection = get_db_connection()


            existing_user = connection.execute(
                """
                SELECT id
                FROM users
                WHERE LOWER(username) = LOWER(?)
                """,
                (username,)
            ).fetchone()


            if existing_user:

                return render_template(
                    "register.html",
                    error="Username already exists."
                )


            password_hash = generate_password_hash(
                password
            )


            cursor = connection.execute(
                """
                INSERT INTO users
                (
                    username,
                    password
                )
                VALUES (?, ?)
                """,
                (
                    username,
                    password_hash
                )
            )


            connection.commit()


            user_id = cursor.lastrowid


        except sqlite3.IntegrityError:

            if connection:

                connection.rollback()


            return render_template(
                "register.html",
                error="Username already exists."
            )


        except sqlite3.OperationalError as error:

            if connection:

                connection.rollback()


            print(
                "Registration database error:",
                error
            )


            return render_template(
                "register.html",
                error=(
                    "Database is busy. "
                    "Please try again."
                )
            )


        except Exception as error:

            if connection:

                connection.rollback()


            print(
                "Registration error:",
                error
            )


            return render_template(
                "register.html",
                error=(
                    "Registration failed. "
                    "Please try again."
                )
            )


        finally:

            if connection:

                connection.close()


        # ----------------------------------------------------
        # LOGIN USER AFTER REGISTRATION
        # ----------------------------------------------------

        flask_session.clear()

        flask_session["user_id"] = user_id

        flask_session["username"] = username


        return redirect(
            url_for("home")
        )


    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if "user_id" in flask_session:

        return redirect(
            url_for("home")
        )


    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )


        connection = None


        try:

            connection = get_db_connection()


            user = connection.execute(
                """
                SELECT *
                FROM users
                WHERE LOWER(username) = LOWER(?)
                """,
                (username,)
            ).fetchone()


        except sqlite3.OperationalError as error:

            print(
                "Login database error:",
                error
            )


            return render_template(
                "login.html",
                error=(
                    "Database is busy. "
                    "Please try again."
                )
            )


        finally:

            if connection:

                connection.close()


        if not user:

            return render_template(
                "login.html",
                error="Invalid username or password."
            )


        try:

            password_correct = (
                check_password_hash(
                    user["password"],
                    password
                )
            )

        except Exception:

            password_correct = False


        if not password_correct:

            return render_template(
                "login.html",
                error="Invalid username or password."
            )


        # ----------------------------------------------------
        # LOGIN SUCCESS
        # ----------------------------------------------------

        flask_session.clear()

        flask_session["user_id"] = user["id"]

        flask_session["username"] = user["username"]


        next_page = request.args.get(
            "next"
        )


        if (
            next_page
            and
            next_page.startswith("/")
        ):

            return redirect(
                next_page
            )


        return redirect(
            url_for("home")
        )


    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    flask_session.clear()

    # Home requires login, so go to login page.

    return redirect(
        url_for("login")
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
@login_required
def home():

    categories = {

        "🔥 Trending This Week":
            get_tmdb_movies(
                "trending/movie/week"
            ),


        "🎬 Popular Movies":
            get_tmdb_movies(
                "movie/popular"
            ),


        "⭐ Top Rated":
            get_tmdb_movies(
                "movie/top_rated"
            ),


        "😂 Comedy":
            get_tmdb_movies(
                "discover/movie",
                {
                    "with_genres": 35
                }
            ),


        "💥 Action":
            get_tmdb_movies(
                "discover/movie",
                {
                    "with_genres": 28
                }
            ),


        "❤️ Romance":
            get_tmdb_movies(
                "discover/movie",
                {
                    "with_genres": 10749
                }
            ),


        "👻 Horror":
            get_tmdb_movies(
                "discover/movie",
                {
                    "with_genres": 27
                }
            ),


        "🕵️ Thriller":
            get_tmdb_movies(
                "discover/movie",
                {
                    "with_genres": 53
                }
            ),


        "🌍 Adventure":
            get_tmdb_movies(
                "discover/movie",
                {
                    "with_genres": 12
                }
            )

    }


    return render_template(
        "index.html",
        categories=categories
    )


# ============================================================
# RECOMMEND PAGE
# ============================================================

@app.route(
    "/recommend",
    methods=["GET", "POST"]
)
@login_required
def recommend_page():

    if request.method == "POST":

        movie_name = request.form.get(
            "movie_name",
            ""
        ).strip()

    else:

        movie_name = request.args.get(
            "movie_name",
            ""
        ).strip()


    # --------------------------------------------------------
    # EMPTY SEARCH
    # --------------------------------------------------------

    if not movie_name:

        return render_template(
            "recommend.html",
            recommendations=[],
            movie_name="",
            corrected_movie_name="",
            selected_movie=None
        )


    print(
        "Movie searched:",
        movie_name
    )


    # --------------------------------------------------------
    # SEARCH TMDB
    # --------------------------------------------------------

    tmdb_movie = search_tmdb_movie(
        movie_name
    )


    if not tmdb_movie:

        return render_template(
            "recommend.html",
            recommendations=[],
            movie_name=movie_name,
            corrected_movie_name=movie_name,
            selected_movie=None
        )


    movie_id = tmdb_movie.get(
        "id"
    )


    corrected_movie_name = tmdb_movie.get(
        "title",
        movie_name
    )


    # --------------------------------------------------------
    # SAVE SEARCH HISTORY
    # --------------------------------------------------------

    add_search_history(
        corrected_movie_name
    )


    # --------------------------------------------------------
    # SELECTED MOVIE
    # --------------------------------------------------------

    selected_movie = (
        get_tmdb_movie_details(
            movie_id
        )
    )


    if selected_movie:

        selected_movie["favorite"] = (
            is_favorite(
                corrected_movie_name
            )
        )


    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = recommend(
        corrected_movie_name
    )


    return render_template(

        "recommend.html",

        recommendations=
            recommendations,

        movie_name=
            movie_name,

        corrected_movie_name=
            corrected_movie_name,

        selected_movie=
            selected_movie

    )


# ============================================================
# MOVIE DETAILS
# ============================================================

@app.route(
    "/movie/<int:movie_id>"
)
@login_required
def movie_details(movie_id):

    selected_movie = (
        get_tmdb_movie_details(
            movie_id
        )
    )


    if not selected_movie:

        return redirect(
            url_for("home")
        )


    selected_movie["favorite"] = (
        is_favorite(
            selected_movie["title"]
        )
    )


    return render_template(

        "recommend.html",

        recommendations=[],

        movie_name=
            selected_movie["title"],

        corrected_movie_name=
            selected_movie["title"],

        selected_movie=
            selected_movie

    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
@login_required
def history():

    connection = None

    try:

        connection = get_db_connection()


        history_data = connection.execute(
            """
            SELECT *
            FROM search_history
            WHERE user_id = ?
            ORDER BY searched_at DESC
            """,
            (
                current_user_id(),
            )
        ).fetchall()


    finally:

        if connection:

            connection.close()


    return render_template(
        "history.html",
        history=history_data
    )


# ============================================================
# DELETE ONE HISTORY ITEM
# ============================================================

@app.route(
    "/delete_history/<int:history_id>",
    methods=["POST"]
)
@login_required
def delete_history(history_id):

    connection = None

    try:

        connection = get_db_connection()


        connection.execute(
            """
            DELETE FROM search_history
            WHERE id = ?
            AND user_id = ?
            """,
            (
                history_id,
                current_user_id()
            )
        )


        connection.commit()


    finally:

        if connection:

            connection.close()


    return redirect(
        url_for("history")
    )


# ============================================================
# CLEAR ALL HISTORY
# ============================================================

@app.route(
    "/clear_history",
    methods=["POST"]
)
@login_required
def clear_history():

    connection = None

    try:

        connection = get_db_connection()


        connection.execute(
            """
            DELETE FROM search_history
            WHERE user_id = ?
            """,
            (
                current_user_id(),
            )
        )


        connection.commit()


    finally:

        if connection:

            connection.close()


    return redirect(
        url_for("history")
    )


# ============================================================
# FAVORITES PAGE
# ============================================================

@app.route("/favorites")
@login_required
def favorites():

    connection = None

    try:

        connection = get_db_connection()


        favorites_data = connection.execute(
            """
            SELECT *
            FROM favorites
            WHERE user_id = ?
            ORDER BY added_at DESC
            """,
            (
                current_user_id(),
            )
        ).fetchall()


    finally:

        if connection:

            connection.close()


    favorite_movies = []


    for favorite in favorites_data:

        movie_id = favorite["movie_id"]


        if movie_id:

            details = (
                get_tmdb_movie_details(
                    movie_id
                )
            )

        else:

            details = get_movie_details(
                favorite["movie_name"]
            )


        if details:

            details["favorite_id"] = (
                favorite["id"]
            )

            details["favorite"] = True


            favorite_movies.append(
                details
            )


    return render_template(
        "favorites.html",
        favorites=favorite_movies
    )


# ============================================================
# ADD FAVORITE
# ============================================================

@app.route(
    "/add_favorite",
    methods=["POST"]
)
@login_required
def add_favorite():

    movie_name = request.form.get(
        "movie_name",
        ""
    ).strip()


    movie_id = request.form.get(
        "movie_id",
        None
    )


    if not movie_name:

        return redirect(
            url_for(
                "recommend_page"
            )
        )


    connection = None


    try:

        connection = get_db_connection()


        connection.execute(
            """
            INSERT OR IGNORE INTO favorites
            (
                movie_name,
                movie_id,
                user_id
            )
            VALUES (?, ?, ?)
            """,
            (
                movie_name,
                movie_id,
                current_user_id()
            )
        )


        connection.commit()


    except Exception as error:

        if connection:

            connection.rollback()


        print(
            "Favorite error:",
            error
        )


    finally:

        if connection:

            connection.close()


    return redirect(
        request.referrer
        or
        url_for(
            "recommend_page"
        )
    )


# ============================================================
# REMOVE FAVORITE
# ============================================================

@app.route(
    "/remove_favorite/<int:favorite_id>",
    methods=["POST"]
)
@login_required
def remove_favorite(
    favorite_id
):

    connection = None


    try:

        connection = get_db_connection()


        connection.execute(
            """
            DELETE FROM favorites
            WHERE id = ?
            AND user_id = ?
            """,
            (
                favorite_id,
                current_user_id()
            )
        )


        connection.commit()


    finally:

        if connection:

            connection.close()


    return redirect(
        request.referrer
        or
        url_for(
            "favorites"
        )
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )