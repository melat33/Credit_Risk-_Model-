"""
Task 4: Proxy Target Variable Engineering
Creates RFM-based credit risk labels using clustering
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFMTargetCreator:
    """Creates proxy target variable using RFM analysis and clustering"""
    
    def __init__(self, snapshot_date=None, random_state=42):
        """
        Args:
            snapshot_date: Reference date for Recency calculation (default: max date in data)
            random_state: For reproducibility
        """
        self.snapshot_date = snapshot_date
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=random_state)
        self.rfm_stats = None
        
    def calculate_rfm(self, df, customer_col='CustomerId', 
                     amount_col='Amount', date_col='TransactionStartTime'):
        """
        Calculate RFM metrics for each customer
        
        Args:
            df: DataFrame with transaction data
            customer_col: Customer identifier column
            amount_col: Transaction amount column
            date_col: Transaction datetime column
            
        Returns:
            DataFrame with RFM metrics per customer
        """
        logger.info("Calculating RFM metrics...")
        
        # Ensure datetime format
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Set snapshot date if not provided
        if self.snapshot_date is None:
            self.snapshot_date = df[date_col].max()
        
        # Calculate RFM
        rfm = df.groupby(customer_col).agg({
            date_col: lambda x: (self.snapshot_date - x.max()).days,  # Recency
            'TransactionId': 'count',  # Frequency
            amount_col: 'sum'  # Monetary
        }).rename(columns={
            date_col: 'recency',
            'TransactionId': 'frequency',
            amount_col: 'monetary'
        })
        
        # Log negative monetary values (credits)
        negative_monetary = (rfm['monetary'] < 0).sum()
        if negative_monetary > 0:
            logger.warning(f"Found {negative_monetary} customers with negative monetary values")
        
        # Use absolute value for monetary (or handle based on business logic)
        rfm['monetary'] = rfm['monetary'].abs()
        
        self.rfm_stats = rfm.describe()
        logger.info(f"RFM calculated for {len(rfm)} customers")
        
        return rfm
    
    def create_clusters(self, rfm_df):
        """
        Create customer segments using K-Means clustering
        
        Args:
            rfm_df: DataFrame with RFM metrics
            
        Returns:
            DataFrame with cluster assignments and analysis
        """
        logger.info("Creating customer clusters...")
        
        # Scale features
        features = ['recency', 'frequency', 'monetary']
        X_scaled = self.scaler.fit_transform(rfm_df[features])
        
        # Apply K-Means
        rfm_df['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_stats = rfm_df.groupby('cluster')[features].agg(['mean', 'std'])
        
        # Identify high-risk cluster (high recency, low frequency, low monetary)
        cluster_profiles = {}
        for cluster_id in sorted(rfm_df['cluster'].unique()):
            profile = {
                'recency_mean': rfm_df[rfm_df['cluster'] == cluster_id]['recency'].mean(),
                'frequency_mean': rfm_df[rfm_df['cluster'] == cluster_id]['frequency'].mean(),
                'monetary_mean': rfm_df[rfm_df['cluster'] == cluster_id]['monetary'].mean(),
                'size': (rfm_df['cluster'] == cluster_id).sum()
            }
            cluster_profiles[cluster_id] = profile
        
        # Determine which cluster is high-risk
        # High-risk typically: High recency (inactive), low frequency, low monetary
        risk_scores = {}
        for cluster_id, profile in cluster_profiles.items():
            # Higher recency = higher risk, lower frequency = higher risk, lower monetary = higher risk
            risk_score = (
                profile['recency_mean'] / rfm_df['recency'].max() +  # Normalized
                (1 - profile['frequency_mean'] / rfm_df['frequency'].max()) +
                (1 - profile['monetary_mean'] / rfm_df['monetary'].max())
            ) / 3
            risk_scores[cluster_id] = risk_score
        
        # Cluster with highest risk score is high-risk
        high_risk_cluster = max(risk_scores, key=risk_scores.get)
        
        logger.info(f"Cluster analysis complete:")
        for cluster_id, profile in cluster_profiles.items():
            risk_label = "HIGH-RISK" if cluster_id == high_risk_cluster else "low-risk"
            logger.info(f"  Cluster {cluster_id} ({risk_label}): "
                       f"Recency={profile['recency_mean']:.1f}d, "
                       f"Frequency={profile['frequency_mean']:.1f}, "
                       f"Monetary=${profile['monetary_mean']:.0f}, "
                       f"Size={profile['size']} customers")
        
        return rfm_df, high_risk_cluster, cluster_profiles
    
    def create_target_variable(self, rfm_df, high_risk_cluster):
        """
        Create binary target variable
        
        Args:
            rfm_df: DataFrame with cluster assignments
            high_risk_cluster: ID of the high-risk cluster
            
        Returns:
            Series with target variable
        """
        logger.info(f"Creating target variable: Cluster {high_risk_cluster} = high-risk")
        
        # Create binary target
        target = (rfm_df['cluster'] == high_risk_cluster).astype(int)
        
        # Rename for clarity
        target = pd.Series(target, index=rfm_df.index)
        target.name = 'is_high_risk'
        
        logger.info(f"Target distribution:")
        logger.info(f"  Low-risk (0): {(target == 0).sum()} customers "
                   f"({(target == 0).mean()*100:.1f}%)")
        logger.info(f"  High-risk (1): {(target == 1).sum()} customers "
                   f"({(target == 1).mean()*100:.1f}%)")
        
        return target
    
    def validate_proxy(self, rfm_df, target, original_df):
        """
        Validate the proxy target makes business sense
        
        Args:
            rfm_df: RFM features DataFrame
            target: Target variable series
            original_df: Original transaction data
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating proxy target...")
        
        # Merge target with RFM for analysis
        validation_df = rfm_df.copy()
        validation_df['is_high_risk'] = target
        
        # Calculate validation metrics
        validation_metrics = {}
        
        # 1. Compare with fraud data (if available)
        if 'FraudResult' in original_df.columns:
            # Get fraud rate by customer
            customer_fraud = original_df.groupby('CustomerId')['FraudResult'].max()
            validation_df['had_fraud'] = customer_fraud
            
            fraud_by_risk = validation_df.groupby('is_high_risk')['had_fraud'].mean()
            validation_metrics['fraud_rate_low_risk'] = fraud_by_risk.get(0, 0)
            validation_metrics['fraud_rate_high_risk'] = fraud_by_risk.get(1, 0)
            validation_metrics['fraud_risk_ratio'] = (
                validation_metrics['fraud_rate_high_risk'] / 
                validation_metrics['fraud_rate_low_risk'] 
                if validation_metrics['fraud_rate_low_risk'] > 0 else float('inf')
            )
        
        # 2. Compare transaction patterns
        for metric in ['recency', 'frequency', 'monetary']:
            low_risk_mean = validation_df[validation_df['is_high_risk'] == 0][metric].mean()
            high_risk_mean = validation_df[validation_df['is_high_risk'] == 1][metric].mean()
            validation_metrics[f'{metric}_ratio'] = high_risk_mean / low_risk_mean if low_risk_mean > 0 else float('inf')
        
        # 3. Business sense validation
        # High-risk should have higher recency, lower frequency, lower monetary
        validation_metrics['makes_business_sense'] = (
            validation_metrics.get('recency_ratio', 0) > 1 and
            validation_metrics.get('frequency_ratio', float('inf')) < 1 and
            validation_metrics.get('monetary_ratio', float('inf')) < 1
        )
        
        logger.info(f"Validation metrics: {validation_metrics}")
        
        return validation_metrics
    
    def run_pipeline(self, df, save_paths=None):
        """
        Run complete proxy target creation pipeline
        
        Args:
            df: Input transaction DataFrame
            save_paths: Dictionary with paths to save outputs
            
        Returns:
            Tuple: (target_variable, rfm_features, validation_metrics)
        """
        logger.info("Starting proxy target creation pipeline...")
        
        # Step 1: Calculate RFM
        rfm_df = self.calculate_rfm(df)
        
        # Step 2: Create clusters
        rfm_df, high_risk_cluster, cluster_profiles = self.create_clusters(rfm_df)
        
        # Step 3: Create target variable
        target = self.create_target_variable(rfm_df, high_risk_cluster)
        
        # Step 4: Validate
        validation_metrics = self.validate_proxy(rfm_df, target, df)
        
        # Save outputs if paths provided
        if save_paths:
            self.save_outputs(rfm_df, target, save_paths)
        
        logger.info("Proxy target creation complete!")
        
        return target, rfm_df, validation_metrics
    
    def save_outputs(self, rfm_df, target, save_paths):
        """Save intermediate and final outputs"""
        import os
        
        # Create directories if they don't exist
        for path in save_paths.values():
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save RFM features
        if 'rfm_path' in save_paths:
            rfm_df.to_csv(save_paths['rfm_path'])
            logger.info(f"Saved RFM features to {save_paths['rfm_path']}")
        
        # Save target variable
        if 'target_path' in save_paths:
            target.to_csv(save_paths['target_path'])
            logger.info(f"Saved target variable to {save_paths['target_path']}")
        
        # Save combined dataset
        if 'combined_path' in save_paths:
            combined = rfm_df.copy()
            combined['is_high_risk'] = target
            combined.to_csv(save_paths['combined_path'])
            logger.info(f"Saved combined dataset to {save_paths['combined_path']}")


def main():
    """Main execution function"""
    # Example usage
    import sys
    sys.path.append('..')
    
    # Load your processed data
    # df = pd.read_csv('data/processed/cleaned_data.csv')
    
    # Initialize creator
    creator = RFMTargetCreator(random_state=42)
    
    # Define save paths
    save_paths = {
        'rfm_path': 'data/processed/customer_rfm.csv',
        'target_path': 'data/processed/target_variable.csv',
        'combined_path': 'data/processed/labeled_dataset.csv'
    }
    
    # Run pipeline (uncomment when you have data)
    # target, rfm, metrics = creator.run_pipeline(df, save_paths)
    
    print("Proxy target creation module ready!")


if __name__ == "__main__":
    main()