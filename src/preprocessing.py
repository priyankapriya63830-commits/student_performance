def preprocess_data(df):

    df = df.copy()

    print("Columns in dataset:", df.columns)

    feature_cols = [
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Scores"
    ]

    X = df[feature_cols]
    y = df.iloc[:, -1]

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler