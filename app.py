
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Movie Recommendation System")
st.write("Discover movies and get personalized recommendations.")

BASE_DIR = Path(__file__).resolve().parent

MOVIE_INFO_PATH = BASE_DIR / "movie_info.csv"
SIMILARITY_PATH = BASE_DIR / "item_similarity.pkl"
TRAIN_MATRIX_PATH = BASE_DIR / "train_matrix.pkl"
GLOBAL_MEAN_PATH = BASE_DIR / "global_mean.pkl"


# --------------------------------------------------
# LOAD SAVED FILES
# --------------------------------------------------

@st.cache_resource
def load_model():
    required_files = [
        MOVIE_INFO_PATH,
        SIMILARITY_PATH,
        TRAIN_MATRIX_PATH,
        GLOBAL_MEAN_PATH,
    ]

    missing_files = [str(path.name) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Missing required files: " + ", ".join(missing_files)
        )

    movie_info = pd.read_csv(MOVIE_INFO_PATH)
    item_similarity = joblib.load(SIMILARITY_PATH)
    train_matrix = joblib.load(TRAIN_MATRIX_PATH)
    global_mean = joblib.load(GLOBAL_MEAN_PATH)

    return item_similarity, train_matrix, global_mean, movie_info


try:
    item_similarity, train_matrix, global_mean, movie_info = load_model()
except Exception as error:
    st.error(f"Could not load the recommendation files: {error}")
    st.stop()


# --------------------------------------------------
# IDENTIFY MOVIE DATA COLUMNS
# --------------------------------------------------

def find_column(dataframe, possible_names):
    """Find a column without depending on capitalization."""
    lookup = {str(column).strip().lower(): column for column in dataframe.columns}

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


MOVIE_ID_COL = find_column(
    movie_info,
    ["movieId", "movie_id", "movie_id_x", "id", "item_id"],
)

TITLE_COL = find_column(
    movie_info,
    ["title", "movie_title", "name", "movie"],
)

GENRES_COL = find_column(
    movie_info,
    ["genres", "genre", "category"],
)

if TITLE_COL is None:
    st.error(
        "Could not find a movie title column in movie_info.csv. "
        "Expected a column such as 'title'."
    )
    st.write("Available columns:", movie_info.columns.tolist())
    st.stop()

if MOVIE_ID_COL is None:
    # Use movie titles as identifiers if no movie ID column exists.
    MOVIE_ID_COL = TITLE_COL

movie_info = movie_info.dropna(subset=[TITLE_COL]).copy()
movie_info[TITLE_COL] = movie_info[TITLE_COL].astype(str)

# Keep the first occurrence of each title.
movie_info = movie_info.drop_duplicates(subset=[TITLE_COL])
movie_titles = sorted(movie_info[TITLE_COL].tolist())


# --------------------------------------------------
# VALIDATE THE TRAINING MATRIX
# --------------------------------------------------

# This is the fix for:
# AttributeError: 'numpy.float64' object has no attribute 'index'

valid_train_matrix = isinstance(train_matrix, pd.DataFrame)

if valid_train_matrix:
    valid_train_matrix = (
        not train_matrix.empty
        and train_matrix.ndim == 2
        and len(train_matrix.columns) > 0
    )

if not valid_train_matrix:
    st.warning(
        "Personalized recommendations are currently unavailable. "
        "Your train_matrix.pkl file does not contain a valid Pandas "
        "ratings DataFrame."
    )
    st.info(
        "Movie-to-movie recommendations may still work if "
        "item_similarity.pkl is valid. To enable personalized "
        "recommendations, rebuild train_matrix.pkl from your original "
        "user-movie ratings data."
    )


# --------------------------------------------------
# PREPARE ITEM SIMILARITY MATRIX
# --------------------------------------------------

def prepare_similarity(similarity, matrix, movies, movie_id_col):
    """Convert similarity data into a labeled DataFrame when possible."""

    if isinstance(similarity, pd.DataFrame):
        return similarity

    if isinstance(similarity, np.ndarray):
        if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
            return None

        labels = None

        if isinstance(matrix, pd.DataFrame):
            if len(matrix.columns) == similarity.shape[0]:
                labels = list(matrix.columns)

        if labels is None and len(movies) == similarity.shape[0]:
            labels = movies[movie_id_col].tolist()

        if labels is not None:
            return pd.DataFrame(
                similarity,
                index=labels,
                columns=labels,
            )

    return None


similarity_df = prepare_similarity(
    item_similarity,
    train_matrix,
    movie_info,
    MOVIE_ID_COL,
)

if similarity_df is None:
    st.warning(
        "The similarity file could not be interpreted as a labeled "
        "square similarity matrix. Check how item_similarity.pkl was saved."
    )


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def same_id(first, second):
    """Compare IDs safely when one file stores IDs as strings."""
    if pd.isna(first) or pd.isna(second):
        return False
    return str(first).strip() == str(second).strip()


def find_matching_label(labels, value):
    """Return the actual label matching an ID or title."""
    for label in labels:
        if same_id(label, value):
            return label
    return None


def get_movie_id(title):
    rows = movie_info[movie_info[TITLE_COL] == title]

    if rows.empty:
        return None

    return rows.iloc[0][MOVIE_ID_COL]


def get_movie_title(movie_id):
    rows = movie_info[
        movie_info[MOVIE_ID_COL].apply(lambda value: same_id(value, movie_id))
    ]

    if rows.empty:
        # The similarity matrix might use movie titles directly.
        title_rows = movie_info[
            movie_info[TITLE_COL].apply(lambda value: same_id(value, movie_id))
        ]
        if not title_rows.empty:
            return title_rows.iloc[0][TITLE_COL]

        return str(movie_id)

    return rows.iloc[0][TITLE_COL]


def get_movie_details(title):
    rows = movie_info[movie_info[TITLE_COL] == title]

    if rows.empty:
        return {}

    row = rows.iloc[0]
    details = {"Movie": row[TITLE_COL]}

    if GENRES_COL is not None and pd.notna(row[GENRES_COL]):
        details["Genres"] = row[GENRES_COL]

    return details


# --------------------------------------------------
# MOVIE-TO-MOVIE RECOMMENDATIONS
# --------------------------------------------------

def recommend_similar_movies(movie_title, number_of_recommendations=10):
    if similarity_df is None:
        return pd.DataFrame()

    movie_id = get_movie_id(movie_title)

    # The similarity matrix may be indexed by IDs or by movie titles.
    selected_label = find_matching_label(similarity_df.index, movie_id)

    if selected_label is None:
        selected_label = find_matching_label(
            similarity_df.index, movie_title
        )

    if selected_label is None:
        return pd.DataFrame()

    scores = similarity_df.loc[selected_label]

    if isinstance(scores, pd.DataFrame):
        scores = scores.iloc[0]

    scores = pd.to_numeric(scores, errors="coerce").dropna()

    # Remove the selected movie from its own recommendations.
    scores = scores.drop(
        labels=[selected_label],
        errors="ignore",
    )

    scores = scores.sort_values(ascending=False).head(
        number_of_recommendations
    )

    results = []

    for recommended_id, score in scores.items():
        title = get_movie_title(recommended_id)

        results.append(
            {
                "Movie": title,
                "Similarity Score": round(float(score), 4),
            }
        )

    return pd.DataFrame(results).drop_duplicates(subset=["Movie"])


# --------------------------------------------------
# PERSONALIZED RECOMMENDATIONS
# --------------------------------------------------

def recommend_for_user(user_id, number_of_recommendations=10):
    if not isinstance(train_matrix, pd.DataFrame):
        return pd.DataFrame()

    if similarity_df is None:
        return pd.DataFrame()

    actual_user_id = find_matching_label(train_matrix.index, user_id)

    if actual_user_id is None:
        return pd.DataFrame()

    user_ratings = train_matrix.loc[actual_user_id]

    if isinstance(user_ratings, pd.DataFrame):
        user_ratings = user_ratings.iloc[0]

    user_ratings = pd.to_numeric(user_ratings, errors="coerce").dropna()

    # Consider only movies the user has actually rated.
    rated_movies = user_ratings[user_ratings > 0]

    if rated_movies.empty:
        return pd.DataFrame()

    similarity_columns = {
        str(column).strip(): column for column in similarity_df.columns
    }

    weighted_scores = {}
    total_weights = {}

    for candidate_id in train_matrix.columns:
        # Do not recommend a movie already rated by the user.
        if candidate_id in rated_movies.index:
            continue

        matching_candidate = find_matching_label(
            similarity_df.index, candidate_id
        )

        if matching_candidate is None:
            matching_candidate = find_matching_label(
                similarity_df.columns, candidate_id
            )

        if matching_candidate is None:
            continue

        numerator = 0.0
        denominator = 0.0

        for rated_id, rating in rated_movies.items():
            matching_rated = find_matching_label(
                similarity_df.columns, rated_id
            )

            if matching_rated is None:
                matching_rated = find_matching_label(
                    similarity_df.index, rated_id
                )

            if matching_rated is None:
                continue

            try:
                similarity_score = float(
                    similarity_df.loc[matching_candidate, matching_rated]
                )
            except (KeyError, TypeError, ValueError):
                try:
                    similarity_score = float(
                        similarity_df.loc[matching_rated, matching_candidate]
                    )
                except (KeyError, TypeError, ValueError):
                    continue

            if not np.isfinite(similarity_score):
                continue

            numerator += similarity_score * float(rating)
            denominator += abs(similarity_score)

        if denominator > 0:
            weighted_scores[candidate_id] = numerator / denominator
            total_weights[candidate_id] = denominator

    if not weighted_scores:
        return pd.DataFrame()

    ranked = sorted(
        weighted_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:number_of_recommendations]

    results = []

    for movie_id, predicted_rating in ranked:
        results.append(
            {
                "Movie": get_movie_title(movie_id),
                "Predicted Rating": round(float(predicted_rating), 3),
            }
        )

    return pd.DataFrame(results)


# --------------------------------------------------
# STREAMLIT INTERFACE
# --------------------------------------------------

similar_tab, personal_tab, data_tab = st.tabs(
    [
        "🍿 Similar Movies",
        "👤 Personalized Recommendations",
        "🔍 Model Diagnostics",
    ]
)

with similar_tab:
    st.subheader("Find movies similar to your favorite")

    selected_movie = st.selectbox(
        "Choose a movie",
        movie_titles,
        key="similar_movie",
    )

    number_similar = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        key="number_similar",
    )

    if st.button("Find Similar Movies", type="primary"):
        recommendations = recommend_similar_movies(
            selected_movie,
            number_similar,
        )

        if recommendations.empty:
            st.warning(
                "No similar movies were found. Check that the movie IDs "
                "in movie_info.csv match the IDs in item_similarity.pkl."
            )
        else:
            st.success(f"Movies similar to **{selected_movie}**")
            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True,
            )

with personal_tab:
    st.subheader("Get recommendations for a user")

    if not isinstance(train_matrix, pd.DataFrame):
        st.error(
            "Personalized recommendations cannot run because "
            "train_matrix.pkl is not a Pandas DataFrame."
        )
        st.code(
            'train_matrix = joblib.load("train_matrix.pkl")\n'
            'print(type(train_matrix))\n'
            'print(getattr(train_matrix, "shape", None))'
        )
        st.write(
            "Re-create train_matrix.pkl from your original ratings dataset. "
            "Changing this app alone cannot turn a saved scalar into a "
            "real user-movie ratings matrix."
        )

    elif train_matrix.empty:
        st.warning("The training matrix is empty.")

    else:
        user_ids = sorted(
            train_matrix.index.tolist(),
            key=lambda value: str(value),
        )

        selected_user = st.selectbox(
            "Select a user ID",
            user_ids,
            key="personal_user",
        )

        number_personal = st.slider(
            "Number of personalized recommendations",
            min_value=5,
            max_value=20,
            value=10,
            key="number_personal",
        )

        if st.button("Recommend Movies", type="primary"):
            recommendations = recommend_for_user(
                selected_user,
                number_personal,
            )

            if recommendations.empty:
                st.info(
                    "No personalized recommendations could be generated "
                    "for this user. Check that the movie IDs in the "
                    "ratings matrix and similarity matrix match."
                )
            else:
                st.success("Your recommended movies")
                st.dataframe(
                    recommendations,
                    use_container_width=True,
                    hide_index=True,
                )

with data_tab:
    st.subheader("Model Diagnostics")

    st.write("**Movie information file**")
    st.write(f"Rows: {len(movie_info):,}")
    st.write("Columns:", movie_info.columns.tolist())

    st.write("**Similarity matrix**")
    if similarity_df is not None:
        st.write("Type: Pandas DataFrame")
        st.write("Shape:", similarity_df.shape)
        st.write("Sample movie IDs:", list(similarity_df.index[:5]))
    else:
        st.write("Type:", type(item_similarity).__name__)

    st.write("**Training matrix**")
    st.write("Type:", type(train_matrix).__name__)

    if isinstance(train_matrix, pd.DataFrame):
        st.write("Shape:", train_matrix.shape)
        st.write("Sample user IDs:", list(train_matrix.index[:5]))
        st.write("Sample movie IDs:", list(train_matrix.columns[:5]))
    else:
        st.error(
            "The saved training matrix is invalid for personalized "
            "recommendations."
        )
        st.write("Stored value:", repr(train_matrix))
        st.write("Stored shape:", getattr(train_matrix, "shape", None))

    st.write("**Global mean**")
    st.write(repr(global_mean))
