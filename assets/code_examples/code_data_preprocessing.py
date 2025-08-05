"""
Example data preprocessing utilities.
Demonstrates proper error handling and type hints.
"""

import numpy as np
from typing import Union, List, Optional
from sklearn.preprocessing import StandardScaler

def normalize_features(
    data: Union[np.ndarray, List[List[float]]],
    method: str = 'standard',
    exclude_cols: Optional[List[int]] = None
) -> np.ndarray:
    """Normalize features using specified method.
    
    Args:
        data: Input data to normalize
        method: Normalization method ('standard' or 'minmax')
        exclude_cols: Column indices to exclude from normalization
        
    Returns:
        Normalized data array
        
    Raises:
        ValueError: If invalid method specified
    """
    try:
        # Convert to numpy array if needed
        X = np.array(data) if isinstance(data, list) else data.copy()
        
        # Identify columns to normalize
        cols_to_normalize = list(range(X.shape[1]))
        if exclude_cols:
            cols_to_normalize = [i for i in cols_to_normalize if i not in exclude_cols]
            
        if method == 'standard':
            scaler = StandardScaler()
            X[:, cols_to_normalize] = scaler.fit_transform(X[:, cols_to_normalize])
        elif method == 'minmax':
            X[:, cols_to_normalize] = (X[:, cols_to_normalize] - X[:, cols_to_normalize].min(axis=0)) / \
                                    (X[:, cols_to_normalize].max(axis=0) - X[:, cols_to_normalize].min(axis=0))
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return X
        
    except Exception as e:
        raise RuntimeError(f"Error normalizing features: {str(e)}")

def handle_missing_values(
    data: np.ndarray,
    strategy: str = 'mean'
) -> np.ndarray:
    """Handle missing values in data array.
    
    Args:
        data: Input data array
        strategy: Strategy for handling missing values ('mean', 'median', 'zero')
        
    Returns:
        Data array with missing values handled
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
        
    if strategy not in ['mean', 'median', 'zero']:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    # Create copy to avoid modifying original
    X = data.copy()
    
    # Handle missing values
    if strategy == 'mean':
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
    elif strategy == 'median':
        col_median = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_median, inds[1])
    else:  # zero
        X[np.isnan(X)] = 0
        
    return X