import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def engineer_features(self, df):
        # Add technical indicators
        df = add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="volume"
        )

        # Add rolling statistics
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df['close'].rolling(window=window).max()

        # Add lag features
        for lag in [1, 3, 5, 7, 14]:
            df[f'lag_{lag}'] = df['close'].shift(lag)

        # Add price momentum
        df['momentum'] = df['close'] - df['close'].shift(5)

        # Add Fourier transforms for cycle detection
        close_fft = np.fft.fft(df['close'].values)
        fft_df = pd.DataFrame({'fft': close_fft})
        fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
        df['fft_absolute'] = fft_df['absolute']
        df['fft_angle'] = fft_df['angle']

        # Add statistical moments
        df['kurtosis'] = df['close'].rolling(window=30).apply(kurtosis)
        df['skew'] = df['close'].rolling(window=30).apply(skew)

        # Add price change percentage
        df['price_change_pct'] = df['close'].pct_change()

        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Scale features
        df_scaled = self.scaler.fit_transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

        return df_scaled

# Usage:
# fe = FeatureEngineer()
# enhanced_df = fe.engineer_features(original_df)