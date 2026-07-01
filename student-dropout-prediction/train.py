import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def train_from_csv():
    print("--- 1. Loading data from CSV file ---")
    
    # Define the CSV file path
    csv_path = 'model/students_data.csv'
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Error: No file found at {csv_path}. Place the CSV file there first!")
    
    # Read CSV file with pandas
    df = pd.read_csv(csv_path)
    print(f"Dataset successfully loaded. Total rows (students): {len(df)}")

    # 2. Separate features (X) and target (y)
    # 'dropout' column is our target variable to predict
    X = df.drop(columns=['dropout'])
    y = df['dropout']

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- 2. Pipeline processing & model training ---")
    # Pipeline to handle missing values and scaling
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Fill missing values
        ('scaler', StandardScaler()),                 # Normalize the data
        ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
    ])

    # Start training
    pipeline.fit(X_train, y_train)

    # Evaluate model accuracy on test set
    predictions = pipeline.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"\nTraining complete! Model test accuracy: {acc*100:.2f}%")

    # Save the trained model for FastAPI to use
    os.makedirs('model', exist_ok=True)
    with open('model/dropout_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Saved new model to 'model/dropout_pipeline.pkl'.")

if __name__ == '__main__':
    train_from_csv()