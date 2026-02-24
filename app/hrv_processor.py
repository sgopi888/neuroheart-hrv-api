"""
HRV Data Processor Module
Handles IBI calculation, resampling, and windowing
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


class HRVProcessor:
    """Process heart rate data into HRV-ready format"""
    
    def __init__(self, window_size_minutes: int = 15):
        """
        Initialize HRV processor
        
        Args:
            window_size_minutes: Size of aggregation windows in minutes
        """
        self.window_size_minutes = window_size_minutes
    
    def calculate_ibi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate inter-beat intervals (RR intervals) from heart rate
        
        Args:
            df: DataFrame with 'timestamp' and 'bpm' columns
            
        Returns:
            DataFrame with additional 'ibi_ms' and 'delta_sec' columns
        """
        df = df.copy()
        
        # Select only timestamp and bpm
        df = df[['timestamp', 'bpm']].copy()
        
        # Calculate time delta between consecutive beats
        df['delta_sec'] = df['timestamp'].diff().dt.total_seconds()
        
        # Calculate IBI (RR interval) in milliseconds
        # IBI = 60000 / BPM (converts beats per minute to ms between beats)
        df['ibi_ms'] = 60000 / df['bpm']
        
        return df
    
    def resample_to_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data into time windows and aggregate IBI values
        
        Args:
            df: DataFrame with timestamp index and 'bpm', 'ibi_ms' columns
            
        Returns:
            DataFrame with windows containing aggregated metrics
        """
        # Set timestamp as index if not already
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        numeric_cols = ['bpm', 'ibi_ms']
        
        # Resample to 1-minute windows first (mean)
        df_1min = df[numeric_cols].resample('1min').mean()
        
        # Interpolate missing IBI values
        df_1min['ibi_ms'] = df_1min['ibi_ms'].interpolate(method='time')
        
        # Aggregate into specified window size
        window_str = f'{self.window_size_minutes}min'
        df_windows = (
            df_1min.resample(window_str)
            .agg({
                'bpm': 'mean',
                'ibi_ms': list  # Preserve all IBI values for HRV computation
            })
            .reset_index()
        )
        
        # Remove windows with empty IBI lists
        df_windows = df_windows[df_windows['ibi_ms'].apply(lambda x: len(x) > 0)]
        
        # Filter out NaN values from IBI lists
        df_windows['ibi_ms'] = df_windows['ibi_ms'].apply(
            lambda x: [val for val in x if not pd.isna(val)]
        )
        
        return df_windows
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete processing pipeline: calculate IBI and create windows
        
        Args:
            df: Raw heart rate DataFrame
            
        Returns:
            Windowed DataFrame ready for HRV feature extraction
        """
        # Calculate IBI
        df_ibi = self.calculate_ibi(df)
        
        # Resample into windows
        df_windows = self.resample_to_windows(df_ibi)
        
        return df_windows
