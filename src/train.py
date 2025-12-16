# ============================================================================
# SAVE TRAIN/TEST/VALIDATION DATA FILES
# ============================================================================
print("\n" + "="*100)
print("SAVING DATA FILES")
print("="*100)

# Assuming you have the rfm DataFrame with target variable from Task 4
# and you've already split your data for model training

# Check if we have the data splits
print("🔍 Checking available data splits...")

# If you've already split your data in Task 5, use those splits
# If not, let's split the data properly
if 'X_train' in locals() and 'X_test' in locals():
    print("✅ Found existing train/test splits")
    
    # Combine features and target for train, validation, and test
    train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    val_data = pd.concat([X_val, y_val.reset_index(drop=True)], axis=1) if 'X_val' in locals() else None
    test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    
else:
    print("⚠️ No existing splits found. Creating new splits...")
    
    # Load or use your rfm data
    # Assuming rfm DataFrame exists from Task 4
    if 'rfm' in locals() and 'is_high_risk' in rfm.columns:
        print("✅ Using rfm DataFrame from Task 4")
        
        # Select features
        features = ['recency_days', 'transaction_frequency', 'total_monetary_value']
        X = rfm[features]
        y = rfm['is_high_risk']
        
        # Split data: 70% train, 15% validation, 15% test
        from sklearn.model_selection import train_test_split
        
        # First split: 70% train, 30% temp (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Second split: 50% validation, 50% test from temp (15% each of total)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Create DataFrames with features + target
        train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
        val_data = pd.concat([X_val, y_val.reset_index(drop=True)], axis=1)
        test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
        
    else:
        print("❌ No rfm data found. Please complete Task 4 first.")
        # Load from saved file if available
        rfm_path = 'data/processed/customer_rfm_with_target.csv'
        if os.path.exists(rfm_path):
            print(f"Loading from {rfm_path}")
            rfm = pd.read_csv(rfm_path)
            
            # Select features
            features = ['recency_days', 'transaction_frequency', 'total_monetary_value']
            X = rfm[features]
            y = rfm['is_high_risk']
            
            # Split data
            from sklearn.model_selection import train_test_split
            
            # 70% train, 15% validation, 15% test
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            
            train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
            val_data = pd.concat([X_val, y_val.reset_index(drop=True)], axis=1)
            test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
            
        else:
            raise FileNotFoundError(f"Cannot find data file: {rfm_path}")

# Save the data files
data_dir = 'data/processed'
os.makedirs(data_dir, exist_ok=True)

# Save train data
train_path = os.path.join(data_dir, 'train_data.csv')
train_data.to_csv(train_path, index=False)
print(f"✅ Train data saved: {train_path}")
print(f"   • Records: {len(train_data):,}")
print(f"   • High-risk customers: {train_data['is_high_risk'].sum():,} ({train_data['is_high_risk'].mean()*100:.1f}%)")

# Save test data
test_path = os.path.join(data_dir, 'test_data.csv')
test_data.to_csv(test_path, index=False)
print(f"✅ Test data saved: {test_path}")
print(f"   • Records: {len(test_data):,}")
print(f"   • High-risk customers: {test_data['is_high_risk'].sum():,} ({test_data['is_high_risk'].mean()*100:.1f}%)")

# Save validation data if it exists
if val_data is not None:
    val_path = os.path.join(data_dir, 'validation_set.csv')
    val_data.to_csv(val_path, index=False)
    print(f"✅ Validation data saved: {val_path}")
    print(f"   • Records: {len(val_data):,}")
    print(f"   • High-risk customers: {val_data['is_high_risk'].sum():,} ({val_data['is_high_risk'].mean()*100:.1f}%)")

# Also save the full dataset for reference
if 'rfm' in locals():
    full_data_path = os.path.join(data_dir, 'customer_rfm_with_target_full.csv')
    rfm.to_csv(full_data_path, index=False)
    print(f"✅ Full dataset saved: {full_data_path}")

print(f"\n📊 DATA SPLIT SUMMARY:")
print(f"   • Train: {len(train_data):,} records ({len(train_data)/len(rfm)*100:.1f}%)")
if val_data is not None:
    print(f"   • Validation: {len(val_data):,} records ({len(val_data)/len(rfm)*100:.1f}%)")
print(f"   • Test: {len(test_data):,} records ({len(test_data)/len(rfm)*100:.1f}%)")
print(f"   • Total: {len(rfm):,} customers")

# Create a data dictionary file
data_dict = {
    "dataset_name": "Bati Bank Credit Risk Data",
    "created_date": datetime.now().isoformat(),
    "data_files": {
        "train_data.csv": {
            "description": "Training dataset (70% of total)",
            "records": int(len(train_data)),
            "features": list(train_data.columns),
            "target_distribution": {
                "low_risk": int(len(train_data) - train_data['is_high_risk'].sum()),
                "high_risk": int(train_data['is_high_risk'].sum()),
                "high_risk_rate": float(train_data['is_high_risk'].mean() * 100)
            }
        },
        "test_data.csv": {
            "description": "Test dataset (15% of total)",
            "records": int(len(test_data)),
            "features": list(test_data.columns),
            "target_distribution": {
                "low_risk": int(len(test_data) - test_data['is_high_risk'].sum()),
                "high_risk": int(test_data['is_high_risk'].sum()),
                "high_risk_rate": float(test_data['is_high_risk'].mean() * 100)
            }
        }
    },
    "features_description": {
        "recency_days": "Days since last transaction (higher = more risky)",
        "transaction_frequency": "Number of transactions (lower = more risky)",
        "total_monetary_value": "Total transaction amount (lower = more risky)",
        "is_high_risk": "Target variable (1 = high risk, 0 = low risk)"
    }
}

if val_data is not None:
    data_dict["data_files"]["validation_set.csv"] = {
        "description": "Validation dataset (15% of total)",
        "records": int(len(val_data)),
        "features": list(val_data.columns),
        "target_distribution": {
            "low_risk": int(len(val_data) - val_data['is_high_risk'].sum()),
            "high_risk": int(val_data['is_high_risk'].sum()),
            "high_risk_rate": float(val_data['is_high_risk'].mean() * 100)
        }
    }

data_dict_path = os.path.join(data_dir, 'data_dictionary.json')
with open(data_dict_path, 'w') as f:
    json.dump(data_dict, f, indent=4)

print(f"\n📋 Data dictionary saved: {data_dict_path}")

# Now update your training script to use these files
print(f"\n🔄 Updating training script to use saved data files...")

# Read the existing training script
with open(script_path, 'r', encoding='utf-8') as f:
    script_content = f.read()

# Update the default data path in the training script
updated_script = script_content.replace(
    "def load_and_preprocess_data(data_path):",
    "def load_and_preprocess_data(data_path='data/processed/customer_rfm_with_target_full.csv'):"
)

# Also update the argparser default
updated_script = updated_script.replace(
    "    parser.add_argument('--data_path', type=str, required=True,",
    "    parser.add_argument('--data_path', type=str, default='data/processed/customer_rfm_with_target_full.csv',"
)

# Save the updated script
with open(script_path, 'w', encoding='utf-8') as f:
    f.write(updated_script)

print(f"✅ Training script updated to use default data path")

print(f"\n" + "="*60)
print("DATA FILES CREATED SUCCESSFULLY!")
print("="*60)
print(f"📁 Data directory: {data_dir}")
print(f"├── train_data.csv ({len(train_data):,} records)")
print(f"├── test_data.csv ({len(test_data):,} records)")
if val_data is not None:
    print(f"├── validation_set.csv ({len(val_data):,} records)")
print(f"├── customer_rfm_with_target_full.csv ({len(rfm):,} records)")
print(f"└── data_dictionary.json (metadata)")
print("="*60)