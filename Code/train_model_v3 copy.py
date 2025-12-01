import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore') # Untuk menekan peringatan yang mungkin muncul dari GridSearchCV

# --- Pengaturan Path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'Data', 'anemia.csv')
output_dir = os.path.join(script_dir, '..', 'result_n_analys')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    # Load the dataset
    data = pd.read_csv(data_path)
    print(f"Data loaded successfully. Initial rows: {data.shape[0]}")

    # 1. PENCEGAHAN KEJADIAN SEBELUMNYA: Hapus Duplikat
    initial_rows = data.shape[0]
    data.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - data.shape[0]
    print(f"Duplicates removed: {duplicates_removed}. Clean rows: {data.shape[0]}")

    # 2. FEATURE ENGINEERING (Pembobotan Fitur yang Lemah)
    # Tujuan: Menciptakan fitur gabungan yang menonjolkan MCH, MCV, dan MCHC

    # Fitur 1: Rata-Rata Sel Darah Merah (Menjaga nilai fitur lemah secara kolektif)
    data['Mean_RCF'] = (data['MCHC'] + data['MCV'] + data['MCH']) / 3

    # Fitur 2: Indeks Perbedaan Hemoglobin (Mungkin penting untuk klasifikasi)
    data['Hb_MCH_Ratio'] = data['Hemoglobin'] / data['MCH']
    
    print("New features 'Mean_RCF' and 'Hb_MCH_Ratio' created.")

    # Split data into features and target
    X = data.drop('Result', axis=1)
    y = data['Result']

    # Split data into Training/Validation and Test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. PENYESUAIAN HYPERPARAMETER (Memaksa Random Forest menggunakan fitur beragam)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        # Mendorong model untuk memilih subset fitur (termasuk fitur baru yang lemah)
        # 3 = 3 fitur dipilih secara acak pada setiap split, membatasi dominasi Hemoglobin
        'max_features': [2, 3, 'sqrt'] 
    }

    # Inisialisasi GridSearchCV object
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid,
                               cv=5, 
                               n_jobs=-1,
                               scoring='f1',
                               verbose=1)

    # Fit the grid search (Training & Validation)
    print("\nStarting GridSearchCV with new features and max_features tuning...")
    grid_search.fit(X_train_val, y_train_val)

    best_model = grid_search.best_estimator_
    
    # --- EVALUASI AKHIR PADA TEST SET ---
    y_pred_test = best_model.predict(X_test)

    # Evaluate the model on the TEST set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)

    print(f'\nBest Parameters found via CV: {grid_search.best_params_}')
    print(f'Model Accuracy on FINAL TEST SET: {accuracy_test:.4f}')
    print(f'F1 Score on FINAL TEST SET: {f1_test:.4f}')

    # Save results
    with open(os.path.join(output_dir, 'f1_score_final_test_v4.txt'), 'w') as f:
        f.write(f'Best Parameters (from CV): {grid_search.best_params_}\n')
        f.write(f'Final Test Set Accuracy: {accuracy_test:.4f}\n')
        f.write(f'Final Test Set F1 Score: {f1_test:.4f}\n')

    # Generate and save confusion matrix
    cm_test = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['No Anemia', 'Anemia'], yticklabels=['No Anemia', 'Anemia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Final Test Set V4)')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_final_test_v4.png'))

    print(f"\nFinal model evaluation saved in {output_dir}")
    joblib.dump(best_model, 'anemia_model_tuned_v4.pkl')
    print("Tuned model trained and saved as anemia_model_tuned_v4.pkl")

except FileNotFoundError:
    print(f"\n❌ ERROR: File not found. Pastikan 'anemia.csv' ada di direktori '../Data'.")
except Exception as e:
    print(f"\n❌ AN ERROR OCCURRED: {e}")