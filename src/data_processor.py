import pandas as pd

def prepare_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    return data['Close']