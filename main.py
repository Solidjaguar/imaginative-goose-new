import json
from ultra_advanced_gold_predictor import fetch_gold_data, prepare_data, train_model, predict_price

def main():
    gold_data = fetch_gold_data()
    
    if gold_data.empty:
        return json.dumps({"error": "Failed to fetch gold data"})
    
    prepared_data = prepare_data(gold_data)
    model = train_model(prepared_data)
    prediction_data = predict_price(model, prepared_data)
    
    result = {
        'latest_price': float(prepared_data.iloc[-1]),
        'latest_date': prepared_data.index[-1].strftime('%Y-%m-%d'),
        'prediction_data': prediction_data,
    }
    
    return json.dumps(result, default=str)

if __name__ == "__main__":
    print(main())