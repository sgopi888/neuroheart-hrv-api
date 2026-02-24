"""
Weekly Pattern Analysis Module
Analyzes HRV patterns across hours and weekdays
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class WeeklyAnalyzer:
    """Analyze weekly and hourly HRV patterns"""
    
    def __init__(self):
        """Initialize weekly analyzer"""
        self.weekday_names = [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 
            'Friday', 'Saturday', 'Sunday'
        ]
    
    def prepare_temporal_features(self, df_hrv: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features (hour, weekday) to HRV DataFrame
        
        Args:
            df_hrv: DataFrame with HRV metrics and timestamp
            
        Returns:
            DataFrame with added temporal columns
        """
        df = df_hrv.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.dayofweek
        df['weekday_name'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        df['is_weekend'] = df['weekday'].isin([5, 6])
        
        return df
    
    def get_hourly_patterns(self, df_hrv: pd.DataFrame) -> pd.DataFrame:
        """
        Compute hourly HRV patterns averaged across all days
        
        Args:
            df_hrv: HRV DataFrame with temporal features
            
        Returns:
            DataFrame with hourly statistics
        """
        df = self.prepare_temporal_features(df_hrv)
        
        hourly_stats = df.groupby('hour').agg({
            'rmssd': ['mean', 'std', 'min', 'max', 'count'],
            'sdnn': ['mean', 'std'],
            'mean_hr': ['mean', 'std'],
            'lf_hf_ratio': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        hourly_stats.columns = ['_'.join(col).strip('_') for col in hourly_stats.columns.values]
        
        return hourly_stats
    
    def get_best_hrv_hours(self, df_hrv: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Find hours with highest RMSSD (best HRV)
        
        Args:
            df_hrv: HRV DataFrame with temporal features
            top_n: Number of top hours to return
            
        Returns:
            DataFrame with top hours sorted by RMSSD
        """
        df = self.prepare_temporal_features(df_hrv)
        hourly_avg = df.groupby('hour')['rmssd'].mean().reset_index()
        hourly_avg = hourly_avg.sort_values('rmssd', ascending=False).head(top_n)
        hourly_avg.columns = ['hour', 'avg_rmssd']
        
        return hourly_avg
    
    def get_worst_hrv_hours(self, df_hrv: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Find hours with lowest RMSSD (worst HRV)
        
        Args:
            df_hrv: HRV DataFrame with temporal features
            top_n: Number of bottom hours to return
            
        Returns:
            DataFrame with worst hours sorted by RMSSD
        """
        df = self.prepare_temporal_features(df_hrv)
        hourly_avg = df.groupby('hour')['rmssd'].mean().reset_index()
        hourly_avg = hourly_avg.sort_values('rmssd', ascending=True).head(top_n)
        hourly_avg.columns = ['hour', 'avg_rmssd']
        
        return hourly_avg
    
    def get_best_hrv_hours_per_weekday(self, df_hrv: pd.DataFrame) -> Dict:
        """
        Find the best hour for each weekday
        
        Args:
            df_hrv: HRV DataFrame with temporal features
            
        Returns:
            Dictionary mapping weekday to best hour info
        """
        df = self.prepare_temporal_features(df_hrv)
        
        results = {}
        for weekday in range(7):
            df_day = df[df['weekday'] == weekday]
            if len(df_day) > 0:
                best_hour = df_day.groupby('hour')['rmssd'].mean().idxmax()
                best_rmssd = df_day.groupby('hour')['rmssd'].mean().max()
                results[self.weekday_names[weekday]] = {
                    'hour': int(best_hour),
                    'avg_rmssd': float(best_rmssd)
                }
        
        return results
    
    def get_worst_hrv_hours_per_weekday(self, df_hrv: pd.DataFrame) -> Dict:
        """
        Find the worst hour for each weekday
        
        Args:
            df_hrv: HRV DataFrame with temporal features
            
        Returns:
            Dictionary mapping weekday to worst hour info
        """
        df = self.prepare_temporal_features(df_hrv)
        
        results = {}
        for weekday in range(7):
            df_day = df[df['weekday'] == weekday]
            if len(df_day) > 0:
                worst_hour = df_day.groupby('hour')['rmssd'].mean().idxmin()
                worst_rmssd = df_day.groupby('hour')['rmssd'].mean().min()
                results[self.weekday_names[weekday]] = {
                    'hour': int(worst_hour),
                    'avg_rmssd': float(worst_rmssd)
                }
        
        return results
    
    def get_most_stressful_weekdays(self, df_hrv: pd.DataFrame) -> pd.DataFrame:
        """
        Rank weekdays by stress level (lowest RMSSD)
        
        Args:
            df_hrv: HRV DataFrame with temporal features
            
        Returns:
            DataFrame with weekdays ranked by average RMSSD (lowest first)
        """
        df = self.prepare_temporal_features(df_hrv)
        
        weekday_stats = df.groupby(['weekday', 'weekday_name']).agg({
            'rmssd': 'mean',
            'mean_hr': 'mean',
            'lf_hf_ratio': 'mean'
        }).reset_index()
        
        weekday_stats = weekday_stats.sort_values('rmssd', ascending=True)
        weekday_stats['stress_rank'] = range(1, len(weekday_stats) + 1)
        
        return weekday_stats
    
    def get_workweek_difficulty(self, df_hrv: pd.DataFrame) -> pd.DataFrame:
        """
        Rank workweek days (Mon-Fri) by difficulty (lowest RMSSD = hardest)
        
        Args:
            df_hrv: HRV DataFrame with temporal features
            
        Returns:
            DataFrame with workweek days ranked by difficulty
        """
        df = self.prepare_temporal_features(df_hrv)
        
        # Filter to weekdays only (0-4 = Mon-Fri)
        workweek = df[df['weekday'] < 5]
        
        if len(workweek) == 0:
            return pd.DataFrame()
        
        workweek_stats = workweek.groupby(['weekday', 'weekday_name']).agg({
            'rmssd': 'mean',
            'mean_hr': 'mean',
            'sdnn': 'mean'
        }).reset_index()
        
        workweek_stats = workweek_stats.sort_values('rmssd', ascending=True)
        workweek_stats['difficulty_rank'] = range(1, len(workweek_stats) + 1)
        workweek_stats['difficulty_label'] = ['Hardest', 'Hard', 'Medium', 'Easy', 'Easiest'][:len(workweek_stats)]
        
        return workweek_stats
    
    def create_weekly_summary(self, df_hrv: pd.DataFrame) -> Dict:
        """
        Create comprehensive weekly analysis summary
        
        Args:
            df_hrv: HRV DataFrame
            
        Returns:
            Dictionary containing all weekly analyses
        """
        return {
            'hourly_patterns': self.get_hourly_patterns(df_hrv).to_dict('records'),
            'best_hrv_hours': self.get_best_hrv_hours(df_hrv).to_dict('records'),
            'worst_hrv_hours': self.get_worst_hrv_hours(df_hrv).to_dict('records'),
            'best_hours_per_weekday': self.get_best_hrv_hours_per_weekday(df_hrv),
            'worst_hours_per_weekday': self.get_worst_hrv_hours_per_weekday(df_hrv),
            'most_stressful_weekdays': self.get_most_stressful_weekdays(df_hrv).to_dict('records'),
            'workweek_difficulty': self.get_workweek_difficulty(df_hrv).to_dict('records')
        }
