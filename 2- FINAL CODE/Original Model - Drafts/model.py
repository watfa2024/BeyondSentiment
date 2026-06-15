import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('technical_dominant_pr_data.csv', parse_dates=['merge_date'])
df = df.sort_values('merge_date').reset_index(drop=True)

baseline_features = [
    'lines_added', 'lines_deleted', 'code_churn', 'files_changed', 'commits_count', 
    'comment_count', 'review_comment_count', 'participants_count', 'has_tests', 
    'test_coverage_change', 'cyclomatic_avg', 'pr_size_category', 'author_experience', 
    'is_core_contributor', 'author_followers', 'response_time_avg', 'num_approvals', 
    'merge_delay_days', 'has_ci_passed', 'avg_commit_msg_length', 'distinct_langs_changed', 
    'num_reviewers', 'code_owner_involvement', 'review_wait_time', 'num_TODO_FIXME'
]

sentiment_features = [
    'description_sentiment', 'review_sentiment_avg', 'review_sentiment_std', 
    'most_negative_sentiment', 'sentiment_trajectory', 'politeness_score', 
    'uncertainty_score', 'technical_confidence_score', 'reviewer_disagreement_level', 
    'comment_escalation', 'change_justification_clarity', 'emotion_categories'
]

augmented_features = baseline_features + sentiment_features
target = 'future_bug_fixes'

print(f"Dataset: {len(df)} PRs from {df['merge_date'].min().date()} to {df['merge_date'].max().date()}")
print(f"Target distribution: mean={df[target].mean():.3f}, median={df[target].median()}, max={df[target].max()}")

def preprocess(X, cat_cols):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    cat_data = pd.DataFrame(
        encoder.fit_transform(X[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X.index
    )
    X_num = X.drop(cat_cols, axis=1)
    return pd.concat([X_num, cat_data], axis=1)

X_baseline = preprocess(df[baseline_features], ['pr_size_category'])
X_augmented = preprocess(df[augmented_features], ['pr_size_category', 'emotion_categories'])
y = df[target].values

models = {
    "XGBoost": XGBRegressor(objective='count:poisson', eval_metric='poisson-nloglik', random_state=42),
    "LightGBM": LGBMRegressor(objective='poisson', random_state=42, verbose=-1),
    "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
}

param_grids = {
    "XGBoost": {'n_estimators': [100, 200, 400], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]},
    "LightGBM": {'n_estimators': [100, 200, 400], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]},
    "RandomForest": {'n_estimators': [100, 200, 400], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5, 10]},
}

tscv = TimeSeriesSplit(n_splits=5)
results_list = []
predictions = {}
feature_importances = {}

def poisson_deviance(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-10)
    return np.mean(2 * (y_true * np.log(y_true / y_pred + 1e-10) - (y_true - y_pred)))

def evaluate_and_explain(X, y, model_name, model, param_grid, feature_set_name, all_feature_names):
    maes, rmses, devs, spearmans = [], [], [], []
    fold_preds, fold_tests = [], []
    importances, shap_values_all = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, scoring='neg_mean_absolute_error',
                                    random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        pred = best.predict(X_test)

        maes.append(mean_absolute_error(y_test, pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, pred)))
        devs.append(poisson_deviance(y_test, pred))
        spearmans.append(spearmanr(y_test, pred).correlation)

        fold_preds.append(pred)
        fold_tests.append(y_test)

        if hasattr(best, "feature_importances_"):
            importances.append(best.feature_importances_)
        elif "XGB" in model_name:
            importances.append(np.array(list(best.get_booster().get_score(importance_type='gain').values())))
        else:
            importances.append(np.zeros(X.shape[1]))

        if model_name in ["XGBoost", "LightGBM", "RandomForest"]:
            explainer = shap.TreeExplainer(best)
            shap_fold = explainer.shap_values(X_test)
            shap_values_all.append(shap_fold)

    result = {
        "Model": model_name,
        "Features": feature_set_name,
        "MAE": np.mean(maes),
        "RMSE": np.mean(rmses),
        "PoissonDeviance": np.mean(devs),
        "Spearman": np.mean(spearmans),
    }
    results_list.append(result)

    y_test_all = np.concatenate(fold_tests)
    y_pred_all = np.concatenate(fold_preds)

    if importances and len(importances[0]) == len(all_feature_names):
        avg_importance = np.mean([imp for imp in importances if len(imp) == len(all_feature_names)], axis=0)
        top_idx = np.argsort(avg_importance)[-10:][::-1]
        feature_importances[f"{model_name}_{feature_set_name}"] = list(zip(
            [all_feature_names[i] for i in top_idx],
            avg_importance[top_idx]
        ))

    joblib.dump(best, f"{model_name}_{feature_set_name}_model.pkl")

    if shap_values_all:
        shap_values_concat = np.concatenate(shap_values_all, axis=0)
        np.save(f"{model_name}_{feature_set_name}_shap.npy", shap_values_concat)

        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values_concat, X_test, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(f"{model_name}_{feature_set_name}_shap_summary.png")
        plt.close()

    return y_test_all, y_pred_all

print("\nRunning time-series cross-validation...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    y_test_b, y_pred_b = evaluate_and_explain(
        X_baseline, y, name, model, param_grids[name], "Baseline", X_baseline.columns)
    y_test_a, y_pred_a = evaluate_and_explain(
        X_augmented, y, name, model, param_grids[name], "Sentiment+Emotion", X_augmented.columns)

    predictions[name] = (y_test_b, y_pred_b, y_pred_a)

results_df = pd.DataFrame(results_list).round(4)
pivot = results_df.pivot(index="Model", columns="Features", values=["MAE", "RMSE", "Spearman", "PoissonDeviance"])

improvement = pd.DataFrame()
for metric in ["MAE", "RMSE", "PoissonDeviance"]:
    improvement[f"{metric}_improvement_%"] = ((pivot[metric]["Baseline"] - pivot[metric]["Sentiment+Emotion"]) /
                                              pivot[metric]["Baseline"] * 100).round(2)
improvement["Spearman_improvement"] = (pivot["Spearman"]["Sentiment+Emotion"] - pivot["Spearman"]["Baseline"]).round(4)

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(pivot)
print("\n% Improvement (positive = sentiment features helped):")
print(improvement)

best_row = results_df.loc[results_df['MAE'].idxmin()]
print(f"\nBEST MODEL: {best_row['Model']} with {best_row['Features']} features → MAE = {best_row['MAE']:.4f}")

print("\nWilcoxon signed-rank test (is augmented significantly better?)")
print("-" * 60)
for name in models:
    _, pred_base, pred_aug = predictions[name]
    err_base = np.abs(pred_base - predictions[name][0])
    err_aug = np.abs(pred_aug - predictions[name][0])
    stat, p = wilcoxon(err_base, err_aug)
    significant = "YES" if p < 0.05 else "no"
    better = "Sentiment+Emotion" if err_aug.mean() < err_base.mean() else "Baseline"
    print(f"{name:12} p-value = {p:.4f} → Significant improvement? {significant:3} | Winner: {better}")

print("\n" + "="*80)
print("TOP 10 MOST IMPORTANT FEATURES (averaged over folds)")
print("="*80)
for key, imp_list in feature_importances.items():
    model_name, feat_set = key.split("_", 1)
    print(f"\n{model_name} ({feat_set})")
    for feat, score in imp_list[:10]:
        marker = "★" if feat in sentiment_features else ""
        print(f"  {feat:35} {score:8.4f} {marker}")

print("\nSentiment/Emotion features in TOP 10 anywhere?")
found = False
for key, imp_list in feature_importances.items():
    for feat, _ in imp_list[:10]:
        if feat in sentiment_features:
            print(f"  → {feat} is important in {key.split('_',1)[0]}")
            found = True
if not found:
    print("  No sentiment/emotion feature made it into top 10 anywhere.")

results_df.to_csv("model_comparison_detailed_results.csv", index=False)
with open("RESULTS_SUMMARY.txt", "w") as f:
    f.write("PR BUG PREDICTION EXPERIMENT RESULTS\n")
    f.write("="*60 + "\n")
    f.write(str(pivot) + "\n\n")
    f.write(str(improvement) + "\n\n")
    f.write(f"Best model: {best_row['Model']} + {best_row['Features']}\n")
    f.write(f"MAE: {best_row['MAE']:.4f}\n")

print("\nAll results, models, and SHAP plots saved!")
