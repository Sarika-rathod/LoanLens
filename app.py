from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# Fix OpenBLAS memory issue
os.environ["OPENBLAS_NUM_THREADS"] = "1"

app = Flask(__name__)

# =========================
# Load saved objects
# =========================
model = joblib.load("model.pkl")
ohe = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")   # ✅ IMPORTANT

# Columns used for OneHotEncoding
cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category"
]

# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():
    return render_template("index.html")


# =========================
# PREDICT ROUTE
# =========================
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    # Show form
    if request.method == 'GET':
        return render_template("predict.html")

    try:
        # =========================
        # Get form data
        # =========================
        data = request.form
        new_data = pd.DataFrame([data])

        # =========================
        # Convert numeric columns
        # =========================
        numeric_cols = [
            "Applicant_Income", "Coapplicant_Income", "Age",
            "Dependents", "Credit_Score", "Existing_Loans",
            "DTI_Ratio", "Savings", "Collateral_Value",
            "Loan_Amount", "Loan_Term"
        ]

        for col in numeric_cols:
            new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

       
        edu = new_data["Education_Level"].iloc[0]

        if edu == "Graduate":
            new_data["Education_Level"] = 1
        elif edu == "Not Graduate":
            new_data["Education_Level"] = 0
        else:
            return "<h3>Please select valid Education Level</h3>"

       
        encoded = ohe.transform(new_data[cols])

        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(cols)
        )

        
        new_data_final = pd.concat(
            [new_data.drop(columns=cols), encoded_df],
            axis=1
        )

        
        new_data_final = new_data_final.reindex(
            columns=features,
            fill_value=0
        )

       
        scaled_data = scaler.transform(new_data_final)

        
        prediction = model.predict(scaled_data)[0]

        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"

        return render_template("result.html", result=result)

    except Exception as e:
        print("ERROR:", e)
        return f"<h2>Error occurred:</h2><p>{str(e)}</p>"



if __name__ == "__main__":
    app.run(debug=True)