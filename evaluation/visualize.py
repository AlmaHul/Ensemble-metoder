import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Faktiskt pris")
    plt.ylabel("Predikterat pris")
    plt.title("Faktiskt vs. Predikterat huspris")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

def plot_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    plt.barh(X_test.columns[sorted_idx], result.importances_mean[sorted_idx])
    plt.title("Permutation Importance")
    plt.tight_layout()
    plt.show()
