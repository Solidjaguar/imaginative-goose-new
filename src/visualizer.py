import matplotlib.pyplot as plt

def plot_predictions(data, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data.values, label='Historical Data')
    plt.plot(predictions.index, predictions.values, label='Predictions', color='red')
    plt.title('Gold Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('static/gold_predictions.png')
    plt.close()