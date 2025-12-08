import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Pengaturan Path Tetap Sama ---
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

# --- PEMBAGIAN DATA BARU: Training/Validation vs. Test Set ---
# 1. Split data into a combined Training/Validation set and a final Test set (e.g., 80/20)
# 'X_train_val' dan 'y_train_val' akan digunakan untuk training dan hyperparameter tuning (GridSearchCV)
# 'X_test' dan 'y_test' akan digunakan untuk evaluasi akhir (Final Evaluation)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definisikan parameter grid untuk GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inisialisasi GridSearchCV object
# GridSearchCV akan menggunakan Cross-Validation (cv=5) pada data X_train_val/y_train_val
# Bagian ini yang berfungsi sebagai Training dan Validation
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5, # 5-fold Cross-Validation
                           n_jobs=-1,
                           scoring='f1',
                           verbose=2)

# Fit the grid search to the combined Training/Validation data
# Model dilatih dan di-tune di sini
print("Starting GridSearchCV for model training and validation...")
grid_search.fit(X_train_val, y_train_val)

# Dapatkan model terbaik
best_model = grid_search.best_estimator_

# --- EVALUASI AKHIR PADA TEST SET BARU ---
# Make predictions on the **separate TEST set**
y_pred_test = best_model.predict(X_test)

# Evaluate the model on the TEST set
accuracy_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

print(f'\nBest Parameters found via Cross-Validation: {grid_search.best_params_}')
print(f'Model Accuracy on FINAL TEST SET: {accuracy_test:.4f}')
print(f'F1 Score on FINAL TEST SET: {f1_test:.4f}')

# Save F1 score and best parameters
with open(os.path.join(output_dir, 'f1_score_final_test.txt'), 'w') as f:
    f.write(f'Best Parameters (from CV): {grid_search.best_params_}\n')
    f.write(f'Final Test Set Accuracy: {accuracy_test:.4f}\n')
    f.write(f'Final Test Set F1 Score: {f1_test:.4f}\n')

# Generate and save confusion matrix for the FINAL TEST SET
cm_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['No Anemia', 'Anemia'], yticklabels=['No Anemia', 'Anemia'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Final Test Set)')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_final_test.png'))

print(f"\nFinal model evaluation saved in {output_dir}")

# Save the trained model
joblib.dump(best_model, 'anemia_model_tuned.pkl')

print("Tuned model trained and saved as anemia_model_tuned.pkl")