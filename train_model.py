from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib

# Load and process data
data = pd.read_csv('D:\\Download\\archive\\Crash_Analysis_System_CAS_data.csv')
features = ['roadCurvature', 'roadSurface', 'weatherA', 'weatherB', 'speedLimit', 'roadCharacter', 'trafficControl']
data['dangerous'] = data['crashSeverity'].apply(lambda x: 1 if x in ['Fatal Crash', 'Serious Crash'] else 0)
X = pd.get_dummies(data[features], drop_first=True)
y = data['dangerous']

# 1. Handle missing values
imputer = SimpleImputer(strategy='mean')  # Use mean to fill missing values
X_imputed = imputer.fit_transform(X)

# 2. Address data imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

# 3. Save feature column names
columns = X.columns
joblib.dump(columns, 'columns.pkl')

# 4. Create preprocessing pipeline (including imputation and standardization)
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocess and split the dataset
X_preprocessed = preprocessor.fit_transform(X_resampled)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_resampled, test_size=0.2, random_state=42)

# 5. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Output probability distributions on training and testing sets (for model verification)
train_proba = model.predict_proba(X_train)[:, 1]
test_proba = model.predict_proba(X_test)[:, 1]
print("Training set probability distribution (first 10):", train_proba[:10])
print("Testing set probability distribution (first 10):", test_proba[:10])

# 6. Save the model and preprocessor
joblib.dump(model, 'road_safety_model_xgb_best.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
