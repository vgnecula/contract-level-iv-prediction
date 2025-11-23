# lib/models/baselines_binary.py

import numpy as np


class MajorityTrendPred:
    """Predicts based on majority trend in IV changes"""
    
    def __init__(self, feature_columns=None):
        self.name = "MajorityTrend"
        feature_columns = list(feature_columns)
        self.iv_idx = feature_columns.index("ImpliedVolatility")
    
    def predict(self, sequence: np.ndarray) -> int:
        """Single sequence prediction"""
        iv_values = sequence[:, self.iv_idx]
        iv_changes = np.diff(iv_values)
        
        up_days = np.sum(iv_changes >= 0)
        down_days = np.sum(iv_changes < 0)
        total_days = len(iv_changes)
        
        if up_days > down_days:
            return 1
        else:
            return 0


class LastDayPred:
    """Predicts based on last day's IV change"""
    
    def __init__(self, feature_columns=None):
        self.name = "LastDay"
        feature_columns = list(feature_columns)
        self.iv_idx = feature_columns.index("ImpliedVolatility")
    
    def predict(self, sequence: np.ndarray) -> int:
        
        second_last_iv = sequence[-2, self.iv_idx]
        last_iv = sequence[-1, self.iv_idx]
        iv_change = last_iv - second_last_iv
        
        return 1 if iv_change >= 0 else 0


class BiasedRandomPred:
    """Biased random predictions"""
    
    def __init__(self, class1_prob: float = 0.7, seed: int = 42):
        self.name = f"BiasedRandom_{class1_prob:.1f}"
        self.class1_prob = class1_prob
        np.random.seed(seed)
    
    def predict(self, sequence: np.ndarray) -> int:
        """Single sequence prediction (ignores sequence content)"""
        rand_val = np.random.random()
        prediction = 1 if rand_val < self.class1_prob else 0
        return prediction


class RandomPred:
    """Pure random predictions"""
    
    def __init__(self, seed: int = 42):
        self.name = "Random"
        np.random.seed(seed)
    
    def predict(self, sequence: np.ndarray) -> int:
        """Single sequence prediction (ignores sequence content)"""
        val = np.random.random()
        prediction = 1 if val >= 0.5 else 0
        return prediction
