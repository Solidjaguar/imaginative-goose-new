import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(y_true, y_pred, config):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true.values, label='Actual')
    plt.plot(y_true.index, y_pred, label='Predicted')
    plt.title('Gold Price: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('predictions_plot.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = importance.argsort()
    pos = np.arange(sorted_idx.shape[0]) + .5

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(pos, importance[sorted_idx], align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(np.array(feature_names)[sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance for Gold Price Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()