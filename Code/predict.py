import joblib
import pandas as pd

# Load the trained model
model = joblib.load('anemia_model_tuned.pkl')

def predict_anemia(gender, hemoglobin, mch, mchc, mcv):
    """
    Predicts anemia based on user input.
    """
    # Create a dataframe from user input
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Hemoglobin': [hemoglobin],
        'MCH': [mch],
        'MCHC': [mchc],
        'MCV': [mcv]
    })

    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == '__main__':
    # Get user input
    print("Enter the following values to predict anemia:")
    gender = int(input("Gender (0 for female, 1 for male): "))
    hemoglobin = float(input("Hemoglobin: "))
    mch = float(input("MCH: "))
    mchc = float(input("MCHC: "))
    mcv = float(input("MCV: "))

    # Predict and display the result
    result = predict_anemia(gender, hemoglobin, mch, mchc, mcv)

    if result == 1:
        print("\nThe model predicts that the person has anemia.")
    else:
        print("\nThe model predicts that the person does not have anemia.")
