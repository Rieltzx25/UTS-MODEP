"""Bangun preprocessor (ColumnTransformer) untuk numeric + categorical."""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


NUM_FEATURES = [
    "cgpa", "tenth_percentage", "twelfth_percentage", "backlogs",
    "projects_completed", "internships_completed",
    "coding_skill_rating", "communication_skill_rating", "aptitude_skill_rating",
    "hackathons_participated", "certifications_count",
    "stress_level",
    "academic_score", "skill_avg",
]

CAT_FEATURES = [
    "branch", "city_tier", "extracurricular_involvement",
]


def build_preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUM_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES),
    ])
