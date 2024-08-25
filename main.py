import json
from ultra_advanced_gold_predictor import fetch_gold_data, prepare_data, train_model, predict_price

def main():
    gold_data = fetch_gold_data()
    
    if not gold_data:
        return json.dumps({"error": "Failed to fetch gold data"})
    
    X, y = prepare_data(gold_data)
    model = train_model(X, y)
    prediction_data = predict_price(model, gold_data)
    
    result = {
        'latest_price': float(gold_data[0]['price']),
        'latest_date': gold_data[0]['date'],
        'prediction_data': prediction_data,
    }
    
    return json.dumps(result)

if __name__ == "__main__":
    print(main())