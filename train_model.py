import numpy as np
import pickle
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class GestureModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                random_state=42,
                probability=True
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                max_iter=1000,
                random_state=42
            )
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
        
    def load_data(self, data_dir='data'):
        data_files = glob.glob(os.path.join(data_dir, 'gesture_data_*.npz'))
        
        if not data_files:
            raise FileNotFoundError(
                f"No data files found in {data_dir}/. "
                "Please run collect_data.py first to collect training data."
            )
        
        all_data = []
        all_labels = []
        
        print(f"\nLoading data from {len(data_files)} file(s)...")
        for file in data_files:
            loaded = np.load(file)
            all_data.append(loaded['data'])
            all_labels.append(loaded['labels'])
            print(f"  ✓ Loaded {file}: {len(loaded['data'])} samples")
        
        # Combine all data
        X = np.vstack(all_data)
        y = np.hstack(all_labels)
        
        print(f"\nTotal samples loaded: {len(X)}")
        print(f"Feature dimensions: {X.shape[1]}")
        
        # Show distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass distribution:")
        for gesture, count in zip(unique, counts):
            print(f"  {gesture}: {count} samples ({count/len(y)*100:.1f}%)")
        
        return X, y
    
    def train_and_evaluate(self, X, y, test_size=0.2):
        """Train and evaluate all models."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n{'='*60}")
        print("TRAINING MODELS")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train each model
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {name.upper().replace('_', ' ')}...")
            print(f"{'='*60}")
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\nAccuracy: {accuracy*100:.2f}%")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'true_labels': y_test
            }
            
            # Track best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {self.best_model_name.upper().replace('_', ' ')}")
        print(f"Accuracy: {self.best_accuracy*100:.2f}%")
        print(f"{'='*60}\n")
        
        return results
    
    
    def save_model(self, output_dir='models'):
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, 'gesture_classifier.pkl')
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Scaler saved to: {scaler_path}")
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'accuracy': self.best_accuracy,
            'gestures': ['play', 'pause', 'rewind', 'fast_forward']
        }
        
        metadata_path = os.path.join(output_dir, 'model_info.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Best Model: {metadata['model_name']}\n")
            f.write(f"Accuracy: {metadata['accuracy']*100:.2f}%\n")
            f.write(f"Gestures: {', '.join(metadata['gestures'])}\n")
        
        print(f"✓ Model info saved to: {metadata_path}\n")

def main():
    trainer = GestureModelTrainer()
    
    try:
        # Load data
        X, y = trainer.load_data()
        
        # Train and evaluate models
        trainer.train_and_evaluate(X, y)
        trainer.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run 'python main_ml.py' to use the trained model")
        print("  2. The model will classify your gestures in real-time")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run the data collection script first:")
        print("  python collect_data.py")
        print("\nCollect at least 50-100 samples per gesture for best results.\n")

if __name__ == "__main__":
    main()
