"""
HRV Features Extraction Module
Computes time-domain, frequency-domain, and non-linear HRV metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from hrvanalysis import remove_outliers, remove_ectopic_beats
from hrvanalysis import get_time_domain_features, get_frequency_domain_features
from hrvanalysis import get_poincare_plot_features


class HRVFeatureExtractor:
    """Extract HRV features from RR intervals"""
    
    def __init__(
        self, 
        remove_outliers_flag: bool = True,
        ectopic_detection: str = "malik"
    ):
        """
        Initialize feature extractor
        
        Args:
            remove_outliers_flag: Whether to remove outliers
            ectopic_detection: Method for ectopic beat detection ('malik', 'karlsson', 'kamath')
        """
        self.remove_outliers_flag = remove_outliers_flag
        self.ectopic_detection = ectopic_detection
    
    def clean_rr_intervals(self, rr_intervals: List[float]) -> List[float]:
        """
        Clean RR intervals by removing outliers and ectopic beats
        
        Args:
            rr_intervals: List of RR intervals in milliseconds
            
        Returns:
            Cleaned list of RR intervals
        """
        if len(rr_intervals) < 10:
            return rr_intervals
        
        rr_clean = rr_intervals.copy()
        
        # Remove outliers
        if self.remove_outliers_flag:
            rr_clean = remove_outliers(
                rr_intervals=rr_clean,
                low_rri=300,
                high_rri=2000
            )
        
        # Remove ectopic beats
        if len(rr_clean) >= 10:
            rr_clean = remove_ectopic_beats(
                rr_intervals=rr_clean,
                method=self.ectopic_detection
            )
        
        return rr_clean
    
    def extract_time_domain(self, rr_intervals: List[float]) -> Dict:
        """
        Extract time-domain HRV features
        
        Args:
            rr_intervals: Cleaned RR intervals
            
        Returns:
            Dictionary of time-domain metrics
        """
        if len(rr_intervals) < 2:
            return self._empty_time_features()
        
        try:
            features = get_time_domain_features(rr_intervals)
            return features
        except Exception as e:
            print(f"Error extracting time features: {e}")
            return self._empty_time_features()
    
    def extract_frequency_domain(self, rr_intervals: List[float]) -> Dict:
        """
        Extract frequency-domain HRV features
        
        Args:
            rr_intervals: Cleaned RR intervals
            
        Returns:
            Dictionary of frequency-domain metrics
        """
        if len(rr_intervals) < 10:
            return self._empty_frequency_features()
        
        try:
            features = get_frequency_domain_features(rr_intervals)
            return features
        except Exception as e:
            print(f"Error extracting frequency features: {e}")
            return self._empty_frequency_features()
    
    def extract_nonlinear(self, rr_intervals: List[float]) -> Dict:
        """
        Extract non-linear HRV features (Poincaré plot)
        
        Args:
            rr_intervals: Cleaned RR intervals
            
        Returns:
            Dictionary of non-linear metrics
        """
        if len(rr_intervals) < 10:
            return self._empty_nonlinear_features()
        
        try:
            features = get_poincare_plot_features(rr_intervals)
            return features
        except Exception as e:
            print(f"Error extracting nonlinear features: {e}")
            return self._empty_nonlinear_features()
    
    def extract_all_features(self, rr_intervals: List[float]) -> Dict:
        """
        Extract all HRV features from RR intervals
        
        Args:
            rr_intervals: List of RR intervals in milliseconds
            
        Returns:
            Dictionary containing all HRV metrics
        """
        # Clean RR intervals
        rr_clean = self.clean_rr_intervals(rr_intervals)
        
        # Extract features
        time_features = self.extract_time_domain(rr_clean)
        freq_features = self.extract_frequency_domain(rr_clean)
        nonlinear_features = self.extract_nonlinear(rr_clean)
        
        # Combine all features
        all_features = {
            **time_features,
            **freq_features,
            **nonlinear_features,
            'num_rr_intervals': len(rr_clean),
            'num_removed': len(rr_intervals) - len(rr_clean)
        }
        
        return all_features
    
    def process_windows(self, df_windows: pd.DataFrame) -> pd.DataFrame:
        """
        Extract HRV features for all time windows
        
        Args:
            df_windows: DataFrame with 'timestamp' and 'ibi_ms' columns
            
        Returns:
            DataFrame with HRV features for each window
        """
        results = []
        
        for idx, row in df_windows.iterrows():
            timestamp = row['timestamp']
            ibi_list = row['ibi_ms']
            
            # Extract features
            features = self.extract_all_features(ibi_list)
            features['timestamp'] = timestamp
            
            results.append(features)
        
        df_hrv = pd.DataFrame(results)
        
        # Reorder columns to put timestamp first
        cols = ['timestamp'] + [col for col in df_hrv.columns if col != 'timestamp']
        df_hrv = df_hrv[cols]
        
        return df_hrv
    
    @staticmethod
    def _empty_time_features() -> Dict:
        """Return empty time-domain features"""
        return {
            'mean_nni': np.nan,
            'sdnn': np.nan,
            'sdsd': np.nan,
            'rmssd': np.nan,
            'median_nni': np.nan,
            'nni_50': np.nan,
            'pnni_50': np.nan,
            'nni_20': np.nan,
            'pnni_20': np.nan,
            'range_nni': np.nan,
            'cvsd': np.nan,
            'cvnni': np.nan,
            'mean_hr': np.nan,
            'max_hr': np.nan,
            'min_hr': np.nan,
            'std_hr': np.nan
        }
    
    @staticmethod
    def _empty_frequency_features() -> Dict:
        """Return empty frequency-domain features"""
        return {
            'lf': np.nan,
            'hf': np.nan,
            'lf_hf_ratio': np.nan,
            'lfnu': np.nan,
            'hfnu': np.nan,
            'total_power': np.nan,
            'vlf': np.nan
        }
    
    @staticmethod
    def _empty_nonlinear_features() -> Dict:
        """Return empty non-linear features"""
        return {
            'sd1': np.nan,
            'sd2': np.nan,
            'ratio_sd2_sd1': np.nan
        }
