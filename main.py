from data.load_data import load_data
from models.train_model import train_rf
from models.tune_model import tune_rf
from evaluation.evaluate import evaluate
from evaluation.visualize import plot_predictions, plot_feature_importance, plot_permutation_importance

# 1. Ladda data
X_train, X_test, y_train, y_test = load_data()

# 2. Träna grundmodell
rf = train_rf(X_train, y_train)
r2, rmse = evaluate(rf, X_test, y_test)
print("Random Forest - R2:", r2)
print("Random Forest - RMSE:", rmse)

# 3. Tunad modell
best_rf, best_params = tune_rf(X_train, y_train)
print("Bästa hyperparametrar:", best_params)
r2_best, rmse_best = evaluate(best_rf, X_test, y_test)
print("Tuned RF - R2:", r2_best)
print("Tuned RF - RMSE:", rmse_best)

# 4. Visualisering
y_pred = rf.predict(X_test)
plot_predictions(y_test, y_pred)
plot_feature_importance(rf, X_train.columns)
plot_permutation_importance(rf, X_test, y_test)
