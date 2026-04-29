"""
Credit Risk Prediction Script
Usage: python predict.py --age 25 --utilization 0.85 --past_due_90 1 --income 35000
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import os

def load_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_path, 'models/saved_models/best_model.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'models/saved_models/scaler.pkl'))
    features = joblib.load(os.path.join(base_path, 'models/saved_models/feature_columns.pkl'))
    threshold = joblib.load(os.path.join(base_path, 'models/saved_models/optimal_threshold.pkl'))
    return model, scaler, features, threshold

def predict_risk(age, utilization, past_due_30, past_due_60, past_due_90, 
                 debt_ratio, income, credit_lines, real_estate, dependents):
    
    model, scaler, features, threshold = load_model()
    
    input_data = {
        'age': age,
        'RevolvingUtilizationOfUnsecuredLines': utilization,
        'NumberOfTime30-59DaysPastDueNotWorse': past_due_30,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': income,
        'NumberOfOpenCreditLinesAndLoans': credit_lines,
        'NumberOfTimes90DaysLate': past_due_90,
        'NumberRealEstateLoansOrLines': real_estate,
        'NumberOfTime60-89DaysPastDueNotWorse': past_due_60,
        'NumberOfDependents': dependents,
    }
    
    input_df = pd.DataFrame([input_data])
    input_df['TotalPastDue'] = past_due_30 + past_due_60 + past_due_90
    input_df['HasDelinquency'] = 1 if input_df['TotalPastDue'].values[0] > 0 else 0
    input_df['SevereDelinquency'] = 1 if past_due_90 > 0 else 0
    input_df['HighUtilization'] = 1 if utilization > 0.8 else 0
    input_df['LogIncome'] = np.log1p(income)
    input_df['AgeSquared'] = age ** 2
    
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[features]
    input_scaled = scaler.transform(input_df)
    prob_default = model.predict_proba(input_scaled)[0, 1]
    prediction = 1 if prob_default >= threshold else 0
    
    return {
        'default_probability': prob_default,
        'prediction': prediction,
        'risk_level': 'High Risk' if prediction == 1 else 'Low Risk'
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict credit default risk')
    parser.add_argument('--age', type=int, required=True)
    parser.add_argument('--utilization', type=float, required=True)
    parser.add_argument('--past_due_30', type=int, default=0)
    parser.add_argument('--past_due_60', type=int, default=0)
    parser.add_argument('--past_due_90', type=int, default=0)
    parser.add_argument('--debt_ratio', type=float, required=True)
    parser.add_argument('--income', type=float, required=True)
    parser.add_argument('--credit_lines', type=int, required=True)
    parser.add_argument('--real_estate', type=int, default=0)
    parser.add_argument('--dependents', type=int, default=0)
    
    args = parser.parse_args()
    
    result = predict_risk(
        age=args.age,
        utilization=args.utilization,
        past_due_30=args.past_due_30,
        past_due_60=args.past_due_60,
        past_due_90=args.past_due_90,
        debt_ratio=args.debt_ratio,
        income=args.income,
        credit_lines=args.credit_lines,
        real_estate=args.real_estate,
        dependents=args.dependents
    )
    
    print("\n" + "=" * 50)
    print("CREDIT RISK ASSESSMENT RESULT")
    print("=" * 50)
    print(f"Default Probability: {result['default_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print("=" * 50)
