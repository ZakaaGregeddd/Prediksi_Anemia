import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths
data_path = os.path.join(script_dir, '..', 'Data', 'anemia.csv')
output_dir = os.path.join(script_dir, '..', 'result_n_analys')

# Create directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the dataset
data = pd.read_csv(data_path)

# Split data into features and target
X = data.drop('Result', axis=1)
y = data['Result']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           scoring='f1',
                           verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Model accuracy after tuning: {accuracy}')
print(f'F1 Score after tuning: {f1}')

# Save F1 score and best parameters
with open(os.path.join(output_dir, 'f1_score.txt'), 'w') as f:
    f.write(f'Best Parameters: {grid_search.best_params_}\n')
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'F1 Score: {f1}\n')

# Generate and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Anemia', 'Anemia'], yticklabels=['No Anemia', 'Anemia'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (After Tuning)')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_tuned.png'))

print(f"Tuned model evaluation saved in {output_dir}")

# Save the trained model
joblib.dump(best_model, 'anemia_model_tuned.pkl')

print("Tuned model trained and saved as anemia_model_tuned.pkl")
