import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChronotypeMLModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def prepare_features(self, df):
        """Extract and engineer features from sleep data"""
        
        features = pd.DataFrame()
        
        # Basic features
        features['bedtime_hour'] = df['bedtime_hour']
        features['wakeup_hour'] = df['wakeup_hour']
        features['sleep_duration'] = df['sleep_duration']
        features['sleep_quality'] = df['sleep_quality']
        features['disturbances'] = df['disturbances']
        features['rem_percentage'] = df['rem_percentage']
        
        # Circular features for time (important for sleep patterns!)
        features['bedtime_sin'] = np.sin(2 * np.pi * features['bedtime_hour'] / 24)
        features['bedtime_cos'] = np.cos(2 * np.pi * features['bedtime_hour'] / 24)
        features['wakeup_sin'] = np.sin(2 * np.pi * features['wakeup_hour'] / 24)
        features['wakeup_cos'] = np.cos(2 * np.pi * features['wakeup_hour'] / 24)
        
        # Interaction features
        features['sleep_efficiency'] = features['sleep_duration'] / (features['disturbances'] + 1)
        features['quality_per_hour'] = features['sleep_quality'] / features['sleep_duration']
        
        # Weekend effect (if available)
        if 'is_weekend' in df.columns:
            features['is_weekend'] = df['is_weekend']
        
        # Rolling statistics (requires user_id for grouping)
        if 'user_id' in df.columns:
            features['bedtime_regularity'] = df.groupby('user_id')['bedtime_hour'].transform('std')
            features['wakeup_regularity'] = df.groupby('user_id')['wakeup_hour'].transform('std')
        else:
            features['bedtime_regularity'] = df['bedtime_hour'].rolling(7, min_periods=1).std()
            features['wakeup_regularity'] = df['wakeup_hour'].rolling(7, min_periods=1).std()
        
        # Fill NaN values
        features = features.fillna(features.median())
        
        self.feature_names = features.columns.tolist()
        return features
    
    def train(self, df, model_type='random_forest'):
        """Train the ML model"""
        
        print("🔧 Preparing features...")
        X = self.prepare_features(df)
        y = self.label_encoder.fit_transform(df['chronotype'])
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"📈 Features used: {', '.join(self.feature_names)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        # Train
        print(f"🤖 Training {model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\n🔄 Cross-validation scores: {cv_scores}")
        print(f"📊 Mean CV accuracy: {cv_scores.mean():.3f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n⭐ Top 5 Important Features:")
            print(importance_df.head(5))
        
        # Save model
        self.save_model()
        
        return accuracy, confusion_matrix(y_test, y_pred)
    
    def predict(self, bedtime, wakeup, sleep_quality=7, disturbances=0, 
                rem_percentage=22, is_weekend=0, bedtime_regularity=1.0, 
                wakeup_regularity=1.0):
        """Predict chronotype for new user"""
        
        if self.model is None:
            self.load_model()
        
        # Calculate derived features
        sleep_duration = (wakeup - bedtime) if wakeup > bedtime else (24 - bedtime + wakeup)
        
        # Create feature array
        features_dict = {
            'bedtime_hour': bedtime,
            'wakeup_hour': wakeup,
            'sleep_duration': sleep_duration,
            'sleep_quality': sleep_quality,
            'disturbances': disturbances,
            'rem_percentage': rem_percentage,
            'bedtime_sin': np.sin(2 * np.pi * bedtime / 24),
            'bedtime_cos': np.cos(2 * np.pi * bedtime / 24),
            'wakeup_sin': np.sin(2 * np.pi * wakeup / 24),
            'wakeup_cos': np.cos(2 * np.pi * wakeup / 24),
            'sleep_efficiency': sleep_duration / (disturbances + 1),
            'quality_per_hour': sleep_quality / sleep_duration,
            'is_weekend': is_weekend,
            'bedtime_regularity': bedtime_regularity,
            'wakeup_regularity': wakeup_regularity
        }
        
        # Convert to DataFrame
        features = pd.DataFrame([features_dict])
        
        # Ensure all features are present
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0
        
        features = features[self.feature_names]
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        chronotype = self.label_encoder.inverse_transform([prediction])[0]
        
        return chronotype, probabilities
    
    def save_model(self):
        """Save model to disk"""
        joblib.dump(self.model, 'models/chronotype_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        print("💾 Model saved successfully!")
    
    def load_model(self):
        """Load model from disk"""
        import os
        if os.path.exists('models/chronotype_model.pkl'):
            self.model = joblib.load('models/chronotype_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            print("📂 Model loaded successfully!")
            return True
        else:
            print("⚠️ No saved model found. Please train the model first.")
            return False

# Create models directory
import os
os.makedirs('models', exist_ok=True)