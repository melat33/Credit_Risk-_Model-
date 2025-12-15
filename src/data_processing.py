# Add to existing data_processing.py

def add_target_variable(processed_df, target_df, customer_col='CustomerId'):
    """
    Merge target variable with processed features
    
    Args:
        processed_df: DataFrame with engineered features
        target_df: DataFrame with target variable
        customer_col: Column to merge on
        
    Returns:
        DataFrame with target variable added
    """
    # Ensure customer ID is in the index for target_df
    if customer_col not in target_df.columns:
        target_df = target_df.reset_index().rename(columns={'index': customer_col})
    
    # Merge target variable
    merged_df = processed_df.merge(
        target_df[[customer_col, 'is_high_risk']],
        on=customer_col,
        how='left'
    )
    
    # Check for missing targets
    missing_targets = merged_df['is_high_risk'].isna().sum()
    if missing_targets > 0:
        print(f"Warning: {missing_targets} customers missing target variable")
        # Optionally fill with default value (e.g., low risk)
        merged_df['is_high_risk'] = merged_df['is_high_risk'].fillna(0)
    
    return merged_df