import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# --- Path Configuration (Same as before) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'Data', 'anemia.csv')
output_dir = os.path.join(script_dir, '..', 'result_n_analys')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    data = pd.read_csv(data_path)
    data.drop_duplicates(inplace=True)
    print(f"Data loaded and cleaned. Rows: {data.shape[0]}")

    # 1. FEATURE ENGINEERING (Ulangi langkah rekayasa fitur)
    data['Mean_RCF'] = (data['MCHC'] + data['MCV'] + data['MCH']) / 3
    data['Hb_MCH_Ratio'] = data['Hemoglobin'] / data['MCH']
    
    # 2. FOKUS UTAMA: HAPUS FITUR DOMINAN (Hemoglobin dan Gender)
    # Kita hanya menggunakan fitur yang sebelumnya 'lemah'
    features_to_drop = ['Result', 'Hemoglobin', 'Gender']
    
    X = data.drop(features_to_drop, axis=1)
    y = data['Result']

    print(f"Features used for training now: {X.columns.tolist()}")

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. PENYESUAIAN HYPERPARAMETER (Dioptimalkan untuk fitur yang lebih sedikit)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, None], # Mengurangi kedalaman untuk menghindari overfitting pada data yang lebih sensitif
        'min_samples_split': [2, 5],
        'max_features': [1.0, 'sqrt'] # Diuji pada semua fitur yang tersisa
    }

    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring='f1', verbose=1)

    print("\nStarting Robustness Test: Training without Hemoglobin and Gender...")
    grid_search.fit(X_train_val, y_train_val)

    best_model = grid_search.best_estimator_
    
    # --- EVALUASI AKHIR PADA TEST SET ---
    y_pred_test = best_model.predict(X_test)

    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)

    print(f'\n--- ROBUSTNESS TEST RESULTS (NO HEMOGLOBIN/GENDER) ---')
    print(f'Best Parameters found via CV: {grid_search.best_params_}')
    print(f'Model Accuracy on FINAL TEST SET: {accuracy_test:.4f}')
    print(f'F1 Score on FINAL TEST SET: {f1_test:.4f}')

    # Save results and model (optional, just for comparison)
    joblib.dump(best_model, 'anemia_model_robustness_test.pkl')

except FileNotFoundError:
    print(f"\nFile not found. Pastikan 'anemia.csv' ada di direktori '../Data'.")
except Exception as e:
    print(f"\nERROR : {e}")