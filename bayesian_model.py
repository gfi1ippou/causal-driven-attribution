import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

# ---- Configuration ----
target_col = 'final.conversions'
expo_cols = [
    'programmatic.impressions','Google.SEM.impressions','TikTok.impressions',
    'SEO.Non.Brand.impressions','facebook.impressions','CRM.impressions',
    'direct sessions','Unassigned sessions','affiliates.impressions'
]
normalized_weights = {
    'programmatic.impressions': 0.135,
    'Google.SEM.impressions': 0.102,
    'TikTok.impressions': 0.140,
    'SEO.Non.Brand.impressions': 0.045,
    'facebook.impressions': 0.089,
    'CRM.impressions': 0.072,
    'direct sessions': 0.092,
    'Unassigned sessions': 0.185,
    'affiliates.impressions': 0.139
}

# Step 1: Prepare X and y from df using new configuration
X_channels = expo_cols
Y_channel = target_col

X_df = df[X_channels].copy()

# Explicitly convert X_channels to numeric, coercing errors
for col in X_channels:
    X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

# Fill any NaNs that might have been introduced by coercion
X_df = X_df.fillna(0)

y = df[Y_channel].values
X = X_df.values
channel_names = X_df.columns.tolist()

# Step 2: Convert normalized weights into priors for betas (must be > 0 for LogNormal)
# Ensure weights are small positive numbers (e.g., scaled)
beta_prior_means = np.array([normalized_weights.get(k, 0.1) for k in X_channels])
beta_prior_means = np.clip(beta_prior_means, 1e-4, None)  # avoid log(0)

# Convert mean to mu for LogNormal using log transformation
log_mu = np.log(beta_prior_means)
sigma_prior = 1.0  # standard deviation in log-space

# Step 3: Initialize cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage for results
cv_results = {
    'fold': [],
    'in_sample_r2': [],
    'in_sample_rmse': [],
    'out_sample_r2': [],
    'out_sample_rmse': [],
    'trace': [],
    'y_train': [],
    'y_test': [],
    'y_pred_train': [],
    'y_pred_test': []
}

print(f"Starting {n_folds}-fold cross-validation...")
print("="*50)

# Step 4: Cross-validation loop
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold_idx + 1}/{n_folds}")
    print("-" * 20)
    
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Build the model for this fold
    with pm.Model() as hierarchical_model:
        # Intercept (can be negative or positive)
        alpha = pm.Normal("alpha", mu=0, sigma=10)

        # Positive coefficients using LogNormal priors
        beta = pm.Lognormal("beta", mu=log_mu, sigma=sigma_prior, shape=X_train.shape[1])

        # Noise
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Regression
        mu = pm.Deterministic("mu", alpha + pm.math.dot(X_train, beta))

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        # Sample
        trace = pm.sample(
            1000,
            tune=1000,
            target_accept=0.9,
            return_inferencedata=True,
            idata_kwargs={
                "log_likelihood": ["y_obs"],
            },
            progressbar=False  # Reduce output clutter
        )
    
    # Get posterior samples for predictions
    alpha_samples = trace.posterior["alpha"].values.flatten()
    beta_samples = trace.posterior["beta"].values.reshape(-1, X_train.shape[1])
    
    # Calculate IN-SAMPLE predictions (training set)
    y_pred_train_samples = []
    for i in range(len(alpha_samples)):
        pred_train = alpha_samples[i] + np.dot(X_train, beta_samples[i])
        y_pred_train_samples.append(pred_train)
    
    y_pred_train_samples = np.array(y_pred_train_samples)
    y_pred_train = np.mean(y_pred_train_samples, axis=0)
    
    # Calculate OUT-OF-SAMPLE predictions (test set)
    y_pred_test_samples = []
    for i in range(len(alpha_samples)):
        pred_test = alpha_samples[i] + np.dot(X_test, beta_samples[i])
        y_pred_test_samples.append(pred_test)
    
    y_pred_test_samples = np.array(y_pred_test_samples)
    y_pred_test = np.mean(y_pred_test_samples, axis=0)
    
    # Calculate IN-SAMPLE metrics (training performance)
    in_sample_r2 = r2_score(y_train, y_pred_train)
    in_sample_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    
    # Calculate OUT-OF-SAMPLE metrics (test performance)
    out_sample_r2 = r2_score(y_test, y_pred_test)
    out_sample_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Store results
    cv_results['fold'].append(fold_idx + 1)
    cv_results['in_sample_r2'].append(in_sample_r2)
    cv_results['in_sample_rmse'].append(in_sample_rmse)
    cv_results['out_sample_r2'].append(out_sample_r2)
    cv_results['out_sample_rmse'].append(out_sample_rmse)
    cv_results['trace'].append(trace)
    cv_results['y_train'].append(y_train)
    cv_results['y_test'].append(y_test)
    cv_results['y_pred_train'].append(y_pred_train)
    cv_results['y_pred_test'].append(y_pred_test)
    
    print(f"In-Sample  -> R²: {in_sample_r2:.3f}, RMSE: {in_sample_rmse:.3f}")
    print(f"Out-Sample -> R²: {out_sample_r2:.3f}, RMSE: {out_sample_rmse:.3f}")
    print(f"Overfitting Check -> R² Diff: {in_sample_r2 - out_sample_r2:.3f}, RMSE Diff: {out_sample_rmse - in_sample_rmse:.3f}")

# Step 5: Summary of cross-validation results
print("\n" + "="*50)
print("CROSS-VALIDATION SUMMARY")
print("="*50)

in_sample_r2_scores = cv_results['in_sample_r2']
in_sample_rmse_scores = cv_results['in_sample_rmse']
out_sample_r2_scores = cv_results['out_sample_r2']
out_sample_rmse_scores = cv_results['out_sample_rmse']

print(f"\nIN-SAMPLE Performance Statistics:")
print(f"  R-squared -> Mean: {np.mean(in_sample_r2_scores):.3f}, Std: {np.std(in_sample_r2_scores):.3f}, Min: {np.min(in_sample_r2_scores):.3f}, Max: {np.max(in_sample_r2_scores):.3f}")
print(f"  RMSE      -> Mean: {np.mean(in_sample_rmse_scores):.3f}, Std: {np.std(in_sample_rmse_scores):.3f}, Min: {np.min(in_sample_rmse_scores):.3f}, Max: {np.max(in_sample_rmse_scores):.3f}")

print(f"\nOUT-OF-SAMPLE Performance Statistics:")
print(f"  R-squared -> Mean: {np.mean(out_sample_r2_scores):.3f}, Std: {np.std(out_sample_r2_scores):.3f}, Min: {np.min(out_sample_r2_scores):.3f}, Max: {np.max(out_sample_r2_scores):.3f}")
print(f"  RMSE      -> Mean: {np.mean(out_sample_rmse_scores):.3f}, Std: {np.std(out_sample_rmse_scores):.3f}, Min: {np.min(out_sample_rmse_scores):.3f}, Max: {np.max(out_sample_rmse_scores):.3f}")

# Calculate overfitting metrics
r2_diff = np.array(in_sample_r2_scores) - np.array(out_sample_r2_scores)
rmse_diff = np.array(out_sample_rmse_scores) - np.array(in_sample_rmse_scores)

print(f"\nOVERFITTING ANALYSIS:")
print(f"  R² Difference (In-Sample - Out-Sample):")
print(f"    Mean: {np.mean(r2_diff):.3f}, Std: {np.std(r2_diff):.3f}")
print(f"    → Positive values indicate overfitting")
print(f"  RMSE Difference (Out-Sample - In-Sample):")
print(f"    Mean: {np.mean(rmse_diff):.3f}, Std: {np.std(rmse_diff):.3f}")
print(f"    → Positive values indicate overfitting")

print(f"\nDetailed Results by Fold:")
print(f"{'Fold':<4} {'In-R²':<6} {'Out-R²':<7} {'R²-Diff':<7} {'In-RMSE':<8} {'Out-RMSE':<9} {'RMSE-Diff':<9}")
print("-" * 60)
for i in range(n_folds):
    r2_diff_fold = in_sample_r2_scores[i] - out_sample_r2_scores[i]
    rmse_diff_fold = out_sample_rmse_scores[i] - in_sample_rmse_scores[i]
    print(f"{i+1:<4} {in_sample_r2_scores[i]:<6.3f} {out_sample_r2_scores[i]:<7.3f} {r2_diff_fold:<7.3f} {in_sample_rmse_scores[i]:<8.3f} {out_sample_rmse_scores[i]:<9.3f} {rmse_diff_fold:<9.3f}")

# Step 6: Train final model on full dataset for coefficient analysis
print("\n" + "="*50)
print("FINAL MODEL (Full Dataset)")
print("="*50)

with pm.Model() as final_model:
    # Intercept (can be negative or positive)
    alpha = pm.Normal("alpha", mu=0, sigma=10)

    # Positive coefficients using LogNormal priors
    beta = pm.Lognormal("beta", mu=log_mu, sigma=sigma_prior, shape=X.shape[1])

    # Noise
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Regression
    mu = pm.Deterministic("mu", alpha + pm.math.dot(X, beta))

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # Sample
    final_trace = pm.sample(
        1000,
        tune=1000,
        target_accept=0.9,
        return_inferencedata=True,
        idata_kwargs={
            "log_likelihood": ["y_obs"],
        }
    )

# Print summary of final model
print("\nFinal Model Summary:")
summary_df = az.summary(final_trace, var_names=["alpha", "beta", "sigma"])
print(summary_df)

# Add channel names to beta coefficients for easier interpretation
print("\nBeta Coefficients by Channel:")
beta_summary = az.summary(final_trace, var_names=["beta"])
for i, channel in enumerate(channel_names):
    row = beta_summary.iloc[i]
    print(f"  {channel}: {row['mean']:.4f} (95% CI: [{row['hdi_2.5%']:.4f}, {row['hdi_97.5%']:.4f}])")

# Final model performance on full dataset
y_pred_final = final_trace.posterior["mu"].mean(axis=(0, 1)).values
print(f"\nFinal Model Performance (Full Dataset):")
print(f"R-squared: {r2_score(y, y_pred_final):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_final)):.3f}")

# Step 7: Model diagnostics
print("\n" + "="*50)
print("MODEL DIAGNOSTICS")
print("="*50)

# Convergence diagnostics
print("\nConvergence Diagnostics (R-hat):")
rhat_summary = az.rhat(final_trace)
print(f"Max R-hat: {rhat_summary.max().values:.3f}")
if rhat_summary.max().values > 1.01:
    print("⚠️  Warning: Some R-hat values > 1.01, indicating potential convergence issues")
else:
    print("✓ All R-hat values < 1.01, indicating good convergence")

# Effective sample size
print("\nEffective Sample Size:")
ess_summary = az.ess(final_trace)
print(f"Min ESS: {ess_summary.min().values:.0f}")
if ess_summary.min().values < 400:
    print("⚠️  Warning: Some ESS values < 400, consider more samples")
else:
    print("✓ All ESS values >= 400")

print("\nModel evaluation complete!")