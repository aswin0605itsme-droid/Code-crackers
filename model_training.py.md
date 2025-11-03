import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib # Used to save your model

# --- 1. Load Data ---
# We use the Pima Diabetes dataset, a public dataset.
# This matches the presentation's example features: Glucose, BMI, Age, etc. 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=column_names)

# Define features (X) and target (y)
feature_cols = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']
X = data[feature_cols] # Features
y = data['class']      # Target (0 = No Diabetes, 1 = Diabetes)

# --- 2. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Create and Train the AI Model ---
# We use RandomForestClassifier, as specified 
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. Evaluate Model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# Note: Our PPT goal was 92%[cite: 62]. This public dataset gives ~75-80%.
# For a real project, we'd use a larger, cleaner dataset.
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- 5. Save the Model ---
# This creates the 'model.pkl' file your app will use.
joblib.dump(model, 'model.pkl')
print("Model saved as 'model.pkl'!")