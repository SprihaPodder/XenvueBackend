import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    df = df.fillna('Unknown')
    return df

def get_data_limitations(df):
    total = len(df)
    missing = df.isnull().sum().sum()
    incomplete_responses = sum(df.isnull().any(axis=1))
    limitations = [
        {"type": "Missing data", "description": f"{missing} total missing values, {incomplete_responses} incomplete responses"},
        {"type": "Sample size", "description": f"n={total} records"},
        {"type": "Duplicate entries", "description": f"{df.duplicated().sum()} duplicates removed"},
    ]
    return limitations

def detect_biases(df):
    # Example: gender imbalance, age distribution, target variable skew
    gender_dist = df['gender'].value_counts(normalize=True).to_dict()
    age_avg = df['age'].mean()
    target_dist = df['target'].value_counts(normalize=True).to_dict()
    bias_report = (
        f"Gender distribution: {gender_dist}. "
        f"Average age: {age_avg:.1f}. "
        f"Target value distribution: {target_dist}."
    )
    return bias_report

def trend_data(df):
    # Example: Count by age group
    bins = [15, 25, 35, 45, 55, 65]
    labels = ["16-25", "26-35", "36-45", "46-55", "56-65"]
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    trend = df['age_group'].value_counts().sort_index()
    return {
        "labels": trend.index.tolist(),
        "values": trend.values.tolist()
    }

def demographics_data(df):
    # Example: Gender
    gender = df['gender'].value_counts()
    return {
        "labels": gender.index.tolist(),
        "values": gender.values.tolist()
    }
