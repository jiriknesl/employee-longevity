"""
Copyright (c) 2024. All rights reserved.
Licensed under the BSD 2-Clause License. See LICENSE file in the project root for full license information.
"""

import pandas as pd
import numpy as np
import pymc as pm
import argparse

def load_data(filepath):
    """Load and validate CSV data"""
    df = pd.read_csv(filepath)
    required_columns = ["employee_id", "months_in_company", "workload_quota", "salary_ratio"]
    if "stayed" not in df.columns:
        df["stayed"] = 1  # Assume current employees for input file
    if "termination_reason" not in df.columns:
        df["termination_reason"] = 0  # Default for current employees
        
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def predict_employee(model, trace, employee_data):
    """Generate predictions for a single employee"""
    with model:
        # Create a "mu_longevity" for the employee
        mu_new = trace.posterior["alpha_longevity"] + \
                 trace.posterior["beta_quota_longevity"] * employee_data["workload_quota"] + \
                 trace.posterior["beta_salary_longevity"] * employee_data["salary_ratio"]

        longevity_samples = np.random.normal(mu_new, trace.posterior["sigma_longevity"])
        predicted_longevity = float(np.mean(longevity_samples))
        longevity_ci_50 = np.maximum(0, np.percentile(longevity_samples, [25, 75]))
        longevity_ci_75 = np.maximum(0, np.percentile(longevity_samples, [12.5, 87.5]))

        # Probability of staying calculation - adjusted to be more sensitive
        logit_p_stay_new = trace.posterior["alpha_stay"] + \
                          trace.posterior["beta_quota_stay"] * employee_data["workload_quota"] + \
                          trace.posterior["beta_salary_stay"] * employee_data["salary_ratio"]
        p_stay_new = 1/(1+np.exp(-logit_p_stay_new))
        prob_leave = 1 - float(np.mean(p_stay_new))

        # More sensitive thresholds for different cases
        is_high_performer = employee_data["workload_quota"] >= 1.2
        is_underpaid = employee_data["salary_ratio"] <= 0.85
        is_underperformer = employee_data["workload_quota"] <= 0.7
        is_overpaid = employee_data["salary_ratio"] >= 1.2
        tenure_months = employee_data["months_in_company"]

        # Adjust leaving probability based on patterns
        if is_high_performer and is_underpaid:
            prob_leave = max(prob_leave, 0.7)  # High risk of leaving for better offer
        elif is_underperformer and is_overpaid and tenure_months > 36:
            prob_leave = max(prob_leave, 0.6)  # High risk of termination
        elif is_underperformer and tenure_months < 12:
            prob_leave = max(prob_leave, 0.5)  # Risk of early termination

        # Reason probabilities if leaves
        if prob_leave > 0.5:
            if is_high_performer and is_underpaid:
                reason_probs = [0.1, 0.1, 0.8]  # High chance of better offer
            elif is_underperformer:
                reason_probs = [0.1, 0.8, 0.1]  # High chance of termination
            else:
                # Use the model's predicted probabilities
                if "alpha_term" in trace.posterior:
                    logits_new = (
                        trace.posterior["alpha_term"].values + 
                        trace.posterior["beta_quota_term"].values * employee_data["workload_quota"] + 
                        trace.posterior["beta_salary_term"].values * employee_data["salary_ratio"]
                    )
                    exp_logits = np.exp(logits_new)
                    reason_probs_calc = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                    reason_probs = np.mean(reason_probs_calc, axis=(0, 1))
                else:
                    reason_probs = [0.4, 0.3, 0.3]
        else:
            reason_probs = [0.9, 0.05, 0.05]  # Likely to stay

        reason_index = int(np.argmax(reason_probs))
        reason_descriptions = {
            0: "N/A (stays)",
            1: "terminated for underperformance",
            2: "left for better offer"
        }
        predicted_reason = reason_descriptions.get(reason_index, "unknown reason")

    return {
        "employee_id": employee_data["employee_id"],
        "predicted_longevity": predicted_longevity,
        "longevity_ci_50_low": longevity_ci_50[0],
        "longevity_ci_50_high": longevity_ci_50[1],
        "longevity_ci_75_low": longevity_ci_75[0],
        "longevity_ci_75_high": longevity_ci_75[1],
        "probability_of_leaving": prob_leave,
        "predicted_reason": predicted_reason,
        "prob_stays": reason_probs[0],
        "prob_underperformance": reason_probs[1],
        "prob_better_offer": reason_probs[2]
    }

def main():
    parser = argparse.ArgumentParser(description='Employee Retention Predictor')
    parser.add_argument('--learning', required=True, help='Path to historical data CSV')
    parser.add_argument('--input', required=True, help='Path to current employees CSV')
    parser.add_argument('--output', required=True, help='Path for output report CSV')
    
    args = parser.parse_args()
    
    # Load training data
    df = load_data(args.learning)
    
    # Build and train model
    with pm.Model() as model:
        # Priors
        alpha_longevity = pm.Normal("alpha_longevity", mu=0, sigma=1)
        beta_quota_longevity = pm.Normal("beta_quota_longevity", mu=0, sigma=1)
        beta_salary_longevity = pm.Normal("beta_salary_longevity", mu=0, sigma=1)
        
        mu_longevity = alpha_longevity + beta_quota_longevity * df["workload_quota"] + beta_salary_longevity * df["salary_ratio"]
        sigma_longevity = pm.Exponential("sigma_longevity", 1.0)
        observed_longevity = pm.Normal("observed_longevity", mu=mu_longevity, sigma=sigma_longevity, observed=df["months_in_company"])
        
        alpha_stay = pm.Normal("alpha_stay", mu=0, sigma=1)
        beta_quota_stay = pm.Normal("beta_quota_stay", mu=0, sigma=1)
        beta_salary_stay = pm.Normal("beta_salary_stay", mu=0, sigma=1)
        
        logit_p_stay = alpha_stay + beta_quota_stay * df["workload_quota"] + beta_salary_stay * df["salary_ratio"]
        p_stay = pm.invlogit(logit_p_stay)
        stayed_obs = pm.Bernoulli("stayed_obs", p=p_stay, observed=df["stayed"])
        
        left_data = df[df["stayed"] == 0]
        if len(left_data) > 0:
            alpha_term = pm.Normal("alpha_term", mu=0, sigma=1, shape=3)
            beta_quota_term = pm.Normal("beta_quota_term", mu=0, sigma=1, shape=3)
            beta_salary_term = pm.Normal("beta_salary_term", mu=0, sigma=1, shape=3)

            quota_left = left_data["workload_quota"].values
            salary_left = left_data["salary_ratio"].values
            logits = alpha_term + beta_quota_term * quota_left[:, None] + beta_salary_term * salary_left[:, None]
            
            p_reason = pm.Deterministic("p_reason", pm.math.softmax(logits, axis=1))
            term_obs = pm.Categorical("term_obs", p_reason, observed=left_data["termination_reason"])

        trace = pm.sample(1000, tune=500, cores=1, chains=1, random_seed=42)

    # Load current employees
    current_employees = load_data(args.input)
    
    # Generate predictions
    results = []
    for _, employee in current_employees.iterrows():
        prediction = predict_employee(model, trace, employee)
        results.append({
            **employee.to_dict(),  # Include original employee data
            **prediction  # Add predictions
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"Analysis complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
