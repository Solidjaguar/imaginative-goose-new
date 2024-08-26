import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def engineer_features(self, df):
        # Add technical indicators
        df = add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="volume"
        )

        # Add rolling statistics
        df['rolling_mean'] = df['close'].rolling(window=14).mean()
        df['rolling_std'] = df['close'].rolling(window=14).std()

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

        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Scale features
        df_scaled = self.scaler.fit_transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

        return df_scaled

# Usage:
# fe = FeatureEngineer()
# enhanced_df = fe.engineer_features(original_df)