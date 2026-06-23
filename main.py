from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model
from src.evaluation import evaluate_model

def main():

    path = r"D:\priya_internship\data\StudentPerformanceFactors.csv"

    data = load_data(path)

    X, y, scaler = preprocess_data(data)

    model, X_test, y_test = train_model(X, y)

    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()