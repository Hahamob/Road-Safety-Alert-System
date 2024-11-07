# Project Overview
This AI-Based Road Safety Alert System aims to enhance driver safety by predicting hazardous road conditions in real-time. Using machine learning, the system identifies potential dangers based on factors such as road surface, weather, and traffic controls, issuing timely alerts to drivers to promote safer driving. The project is designed with a modular structure, separating the frontend interface from the backend API, which hosts the trained machine learning model. This setup allows the system to analyze input data quickly and deliver clear, actionable alerts to drivers, making it a practical, adaptable solution for improving road safety in diverse driving environments.
# Tech Stack
- Programming Language: Python (for model training and backend)
- Frameworks:
  - Flask: for backend API development
  - HTML/CSS/JavaScript: for frontend interface
- Machine Learning Libraries:
  - Scikit-learn: for data preprocessing, model training, and evaluation
  - Imbalanced-learn (SMOTE): for handling data imbalance
- Other Tools:
  - Joblib: for saving and loading models and preprocessors
  - Pandas: for data processing
  - Flask-CORS: for handling cross-origin requests
# Data Processing and Feature Engineering
Data Source
https://www.kaggle.com/datasets/neonninja/nzta-crash-analysis-system-cas
Feature Selection
List selected features such as:
- roadCurvature: curvature of the road
- roadSurface: surface type of the road
- weatherA and weatherB: primary and secondary weather conditions
- speedLimit: speed limit on the road
- roadCharacter: type of road feature
- trafficControl: type of traffic control
Data Preprocessing
- Missing Value Handling: Using mean imputation to handle missing data.
- Data Balancing: Using SMOTE to oversample minority classes and address data imbalance.
# Model Training and Evaluation
Model Selection
Use a RandomForestClassifier, and briefly justify this choice, e.g., effectiveness in binary classification tasks.
Model Training Steps
- Data Splitting: Divide data into training and test sets.
- Feature Standardization: Standardize features to improve model performance.
- Training and Saving: Train the model and save it for future use.
Model Evaluation
- Evaluate using metrics such as accuracy, precision, recall, and F1 score.
- Include results (e.g., confusion matrix, classification report) to showcase performance.
# System Design
System Architecture
- Frontend: User interface allowing data input and alert display.
- Backend: Flask API for data processing and prediction, responding to frontend requests.
Workflow:
1. The user inputs road condition details via the frontend.
2. The frontend sends data to the backend API.
3. The backend processes the data and returns predictions.
4. The frontend displays alerts to inform the driver.
5. Usage Guide
Frontend:
- Open the index.html page, enter road condition details, and click "Get Safety Alert" to receive safety information.
Backend:
- Start the Flask server by running python server.py.
- Access the API at http://127.0.0.1:5000/predict to handle POST requests from the frontend.
Output:
- JSON response containing safety alerts and danger probability.
