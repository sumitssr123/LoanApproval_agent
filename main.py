from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('loan_approval_model.pkl')

app = FastAPI()

# Allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data structure we expect from the frontend
class LoanForm(BaseModel):
    Applicant_ID: float
    Applicant_Income: float
    Coapplicant_Income: float
    Age: float
    Marital_Status: str
    Dependents: float
    Credit_Score: float
    Existing_Loans: float
    DTI_Ratio: float
    Savings: float
    Collateral_Value: float
    Loan_Amount: float
    Loan_Term: float
    Education_Level: str
    Gender: str
    Employment_Status: str
    Loan_Purpose: str
    Property_Area: str
    Employer_Category: str

@app.post("/predict")
def predict_loan(data: LoanForm):
    # Map the text inputs to 1s and 0s just like your Jupyter code did
    input_dict = {
        "Applicant_ID": data.Applicant_ID,
        "Applicant_Income": data.Applicant_Income,
        "Coapplicant_Income": data.Coapplicant_Income,
        "Age": data.Age,
        "Marital_Status": 1 if data.Marital_Status == "Married" else 0,
        "Dependents": data.Dependents,
        "Credit_Score": data.Credit_Score,
        "Existing_Loans": data.Existing_Loans,
        "DTI_Ratio": data.DTI_Ratio,
        "Savings": data.Savings,
        "Collateral_Value": data.Collateral_Value,
        "Loan_Amount": data.Loan_Amount,
        "Loan_Term": data.Loan_Term,
        "Education_Level": 1 if data.Education_Level == "Graduate" else 0,
        "Gender": 1 if data.Gender == "Male" else 0,
        
        # One-Hot Encoding Mappings
        "Employment_Status_Salaried": 1 if data.Employment_Status == "Salaried" else 0,
        "Employment_Status_Self-employed": 1 if data.Employment_Status == "Self-employed" else 0,
        "Employment_Status_Unemployed": 1 if data.Employment_Status == "Unemployed" else 0,
        
        "Loan_Purpose_Car": 1 if data.Loan_Purpose == "Car" else 0,
        "Loan_Purpose_Education": 1 if data.Loan_Purpose == "Education" else 0,
        "Loan_Purpose_Home": 1 if data.Loan_Purpose == "Home" else 0,
        "Loan_Purpose_Personal": 1 if data.Loan_Purpose == "Personal" else 0,
        
        "Property_Area_Semiurban": 1 if data.Property_Area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if data.Property_Area == "Urban" else 0,
        
        "Employer_Category_Government": 1 if data.Employer_Category == "Government" else 0,
        "Employer_Category_MNC": 1 if data.Employer_Category == "MNC" else 0,
        "Employer_Category_Private": 1 if data.Employer_Category == "Private" else 0,
        "Employer_Category_Unemployed": 1 if data.Employer_Category == "Unemployed" else 0,
    }

    # Convert the dictionary into a Pandas DataFrame
    df = pd.DataFrame([input_dict])
    
    # Make prediction (0 or 1)
    prediction = model.predict(df)[0]
    
    # Return the response
    return {
        "status": int(prediction),
        "message": "Loan Approved" if prediction == 1 else "Loan Rejected"
    }