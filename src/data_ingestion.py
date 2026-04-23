import pandas as pd


def load_data():
    features = pd.read_csv("A.csv")
    targets  = pd.read_csv("A_targets.csv")

    df = features.merge(targets, on="Student_ID").drop(columns=["Student_ID"])

    df["academic_score"] = (df["cgpa"]*10 + df["tenth_percentage"] + df["twelfth_percentage"]) / 3
    df["skill_avg"]      = (df["coding_skill_rating"]
                            + df["communication_skill_rating"]
                            + df["aptitude_skill_rating"]) / 3
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {df.shape}")
    print(df.head())
