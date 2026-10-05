import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Load Data
df = pd.read_csv('rock_drillability_data.csv')
df = df.apply(pd.to_numeric, errors='coerce')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Define SVR Pipeline with Feature Scaling
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])

param_grid = {
    'svr__C': [500],
    'svr__epsilon': [0.5],
    'svr__gamma': [0.01],
    'svr__kernel': ['rbf']
}

svr_grid = GridSearchCV(svr_pipeline, param_grid, cv=5)
svr_grid.fit(X_train, y_train)

# 3. Benchmark Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
    'Tuned SVR (Scaled)': svr_grid.best_estimator_
}

results = []
for name, model in models.items():
    if name != 'Tuned SVR (Scaled)':
        model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    results.append({'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae})

# 4. Display Benchmark Results
results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print("\n--- Corrected Model Benchmark ---")
print(results_df.to_string(index=False))

# 5. Save Model Pipeline to Disk
joblib.dump(svr_grid.best_estimator_, 'best_rock_drillability_svr.pkl')
print("\nSaved best model as 'best_rock_drillability_svr.pkl'")

# 6. Generate and Save Final Visualizations
plt.figure(figsize=(10, 4))

# Plot 1: Feature Importance (XGBoost)
plt.subplot(1, 2, 1)
feat_importances = pd.Series(models['XGBoost'].feature_importances_, index=X.columns)
feat_importances.nlargest(8).plot(kind='barh', color='skyblue')
plt.title('XGBoost Feature Importance')
plt.xlabel('Relative Importance')

# Plot 2: Actual vs Predicted (Tuned SVR)
plt.subplot(1, 2, 2)
svr_preds = models['Tuned SVR (Scaled)'].predict(X_test)
plt.scatter(y_test, svr_preds, alpha=0.7, color='crimson')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual DRI')
plt.ylabel('Predicted DRI')
plt.title('SVR Actual vs. Predicted')

plt.tight_layout()
plt.savefig('drillability_final_results.png', dpi=300)
print("Saved plots as 'drillability_final_results.png'")