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

        # Calculate base probability of staying
        logit_p_stay_new = trace.posterior["alpha_stay"] + \
                          trace.posterior["beta_quota_stay"] * employee_data["workload_quota"] + \
                          trace.posterior["beta_salary_stay"] * employee_data["salary_ratio"]
        p_stay_new = 1/(1+np.exp(-logit_p_stay_new))
        base_prob_leave = 1 - float(np.mean(p_stay_new))

        # More nuanced performance and pay categories
        workload = employee_data["workload_quota"]
        salary = employee_data["salary_ratio"]
        tenure_months = employee_data["months_in_company"]

        # Performance levels
        is_exceptional = workload >= 1.3
        is_high_performer = workload >= 1.15
        is_good_performer = workload >= 0.95
        is_underperformer = workload <= 0.75
        is_serious_underperformer = workload <= 0.65

        # Pay levels relative to performance
        expected_salary = 0.9 + (workload - 0.9) * 1.1  # Simple linear relationship
        pay_gap = salary - expected_salary
        is_underpaid = pay_gap <= -0.15
        is_very_underpaid = pay_gap <= -0.25
        is_overpaid = pay_gap >= 0.15
        is_very_overpaid = pay_gap >= 0.25

        # Tenure-based risk factors
        is_new = tenure_months <= 6
        is_established = 12 <= tenure_months <= 36
        is_veteran = tenure_months >= 48

        # Start with base probability and adjust based on patterns
        prob_leave = base_prob_leave

        # Adjust leaving probability based on patterns
        if is_exceptional and is_very_underpaid:
            prob_leave = max(prob_leave, 0.85)
        elif is_high_performer and is_underpaid:
            prob_leave = max(prob_leave, 0.7)
        elif is_serious_underperformer and is_overpaid:
            if is_established:
                prob_leave = max(prob_leave, 0.75)
            elif is_new:
                prob_leave = max(prob_leave, 0.6)
        elif is_underperformer and is_underpaid:
            prob_leave = max(prob_leave, 0.4 + (tenure_months / 100))
        elif is_good_performer and not is_underpaid and is_established:
            prob_leave = min(prob_leave, 0.3)

        # Reason probabilities with more nuance
        if prob_leave > 0.5:
            if (is_high_performer or is_exceptional) and (is_underpaid or is_very_underpaid):
                better_offer_prob = 0.7 + (0.1 if is_exceptional else 0) + (0.1 if is_very_underpaid else 0)
                reason_probs = [0.1, 0.1, better_offer_prob]
            elif is_underperformer or is_serious_underperformer:
                term_prob = 0.6 + (0.2 if is_serious_underperformer else 0) + (0.1 if is_overpaid else 0)
                reason_probs = [0.1, term_prob, 0.9 - term_prob]
            else:
                # Use model predictions but ensure they're meaningful
                if "alpha_term" in trace.posterior:
                    logits_new = (
                        trace.posterior["alpha_term"].values + 
                        trace.posterior["beta_quota_term"].values * workload + 
                        trace.posterior["beta_salary_term"].values * salary
                    )
                    exp_logits = np.exp(logits_new)
                    reason_probs = np.mean(exp_logits / np.sum(exp_logits, axis=-1, keepdims=True), axis=(0, 1))
                else:
                    reason_probs = [0.4, 0.3, 0.3]
        else:
            reason_probs = [0.9, 0.05, 0.05]

        # Adjust confidence intervals based on tenure
        if is_new:
            longevity_ci_50 = np.maximum(0, longevity_ci_50 * 1.3)
            longevity_ci_75 = np.maximum(0, longevity_ci_75 * 1.4)

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
