import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore') 

# --- Pengaturan Path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'Data', 'anemia.csv')
output_dir = os.path.join(script_dir, '..', 'result_n_analys')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Starting KNN Model Training...")

try:
    # 1. Load Data dan Pre-processing
    data = pd.read_csv(data_path)
    data.drop_duplicates(inplace=True)

    # 2. FEATURE ENGINEERING (Menggunakan fitur yang sama seperti sebelumnya)
    data['Mean_RCF'] = (data['MCHC'] + data['MCV'] + data['MCH']) / 3
    data['Hb_MCH_Ratio'] = data['Hemoglobin'] / data['MCH']
    
    # Split data (masih menggunakan semua fitur untuk KNN)
    X = data.drop('Result', axis=1)
    y = data['Result']

    # Split data into Training/Validation and Test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. SCALING/NORMALISASI DATA (PENTING UNTUK KNN)
    # KNN sangat sensitif terhadap skala data karena didasarkan pada jarak.
    # Kita menggunakan StandardScaler untuk menormalisasi data
    scaler = StandardScaler()
    
    # Fit scaler hanya pada data training/validation untuk menghindari data leakage
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    
    # Transform data test
    X_test_scaled = scaler.transform(X_test)

    # 4. TUNING HYPERPARAMETER (Mencari nilai K terbaik)
    # Mencari nilai K ganjil (untuk menghindari seri) dari 1 hingga 20
    param_grid = {
        'n_neighbors': list(range(1, 21, 2)), 
        'weights': ['uniform', 'distance'] # Menggunakan bobot seragam atau berdasarkan jarak
    }

    # Inisialisasi GridSearchCV object
    # Menggunakan f1 score sebagai metrik yang sama dengan Random Forest
    grid_search = GridSearchCV(estimator=KNeighborsClassifier(),
                               param_grid=param_grid,
                               cv=5, 
                               n_jobs=-1,
                               scoring='f1',
                               verbose=1)

    # Fit the grid search (Training & Validation)
    print("\nStarting GridSearchCV for K-Nearest Neighbors...")
    grid_search.fit(X_train_val_scaled, y_train_val)

    best_model = grid_search.best_estimator_
    
    # --- 5. EVALUASI AKHIR PADA TEST SET ---
    y_pred_test = best_model.predict(X_test_scaled)

    # Evaluate the model on the TEST set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)

    print(f'\n--- K-NEAREST NEIGHBORS (KNN) RESULTS ---')
    print(f'Best Parameters found via CV: {grid_search.best_params_}')
    print(f'Model Accuracy on FINAL TEST SET: {accuracy_test:.4f}')
    print(f'F1 Score on FINAL TEST SET: {f1_test:.4f}')

    # Save results
    with open(os.path.join(output_dir, 'f1_score_knn.txt'), 'w') as f:
        f.write(f'KNN MODEL RESULTS\n')
        f.write(f'Best Parameters (from CV): {grid_search.best_params_}\n')
        f.write(f'Final Test Set Accuracy: {accuracy_test:.4f}\n')
        f.write(f'Final Test Set F1 Score: {f1_test:.4f}\n')

    # Generate and save confusion matrix
    cm_test = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', xticklabels=['No Anemia', 'Anemia'], yticklabels=['No Anemia', 'Anemia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (KNN Model)')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_knn.png'))

    print("\nKNN model trained and results saved.")

except FileNotFoundError:
    print(f"\n❌ ERROR: File not found. Pastikan 'anemia.csv' ada di direktori '../Data'.")
except Exception as e:
    print(f"\n❌ AN ERROR OCCURRED: {e}")