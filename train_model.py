import pandas as pd
from data_generator import generate_sleep_data
from ml_model import ChronotypeMLModel

def main():
    print("="*50)
    print("🌙 Sleep Chronotype ML Model Training")
    print("="*50)
    
    # Generate training data
    print("\n📊 Generating training data...")
    df = generate_sleep_data(n_days=200, n_users=4)
    
    print(f"✅ Generated {len(df)} sleep records")
    print(f"📈 Data shape: {df.shape}")
    print(f"\n📊 Chronotype distribution:")
    print(df['chronotype'].value_counts())
    
    # Train model
    print("\n" + "="*50)
    ml_model = ChronotypeMLModel()
    
    # Try both model types
    print("\n🚀 Training Random Forest model...")
    accuracy_rf, cm_rf = ml_model.train(df, model_type='random_forest')
    
    print("\n" + "="*50)
    print("🎉 Training Complete!")
    print(f"✅ Best Model Accuracy: {accuracy_rf*100:.1f}%")
    print("="*50)
    
    # Test prediction
    print("\n🔮 Testing prediction on sample data...")
    test_cases = [
        (22.5, 6.0, "Should be Lion"),
        (23.0, 7.5, "Should be Bear"),
        (1.5, 9.0, "Should be Wolf"),
        (0.0, 6.0, "Should be Dolphin")
    ]
    
    for bedtime, wakeup, expected in test_cases:
        chronotype, probs = ml_model.predict(bedtime, wakeup)
        print(f"  Bedtime: {bedtime:.1f}h, Wakeup: {wakeup:.1f}h → {chronotype} ({expected})")
    
    print("\n💡 Model is ready! Run 'streamlit run app.py' to use the web interface")

if __name__ == "__main__":
    main()