import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

# config/constants
TARGET_COL = "Outcome"
POP_SIZE = 100
ELITE_SIZE = 20
MUT_RATE = 0.15
HIDDEN_SIZE = 32
MAX_GENERATIONS = 1000

# preparation
df = pd.read_csv("data/diabetes.csv")
y = df[TARGET_COL].values
X = df.drop(columns=[TARGET_COL])

# standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

input_size = X_train.shape[1]
output_size = len(np.unique(y))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
def evaluate_individual(model, X_tensor, y_tensor):
    with torch.no_grad():
        pred = model(X_tensor)
        pred_labels = pred.argmax(dim=1)
        acc = (pred_labels == y_tensor).float().mean().item()
        return acc
        
def crossover(p1, p2):
    child = Net()
    with torch.no_grad():
        for (_, param_c), (_, param_p1), (_, param_p2) in zip(child.named_parameters(), p1.named_parameters(), p2.named_parameters()):
            mask = torch.rand_like(param_c) > 0.5
            param_c.copy_(torch.where(mask, param_p1, param_p2))
    return child
        

def mutate(model):
    with torch.no_grad():
        for param in model.parameters():
            mask = torch.rand_like(param) < MUT_RATE
            noise = torch.randn_like(param) / 10
            param[mask] += noise[mask]
            

class HybridEnsembleTrainer:
    def __init__(self, pop_size=POP_SIZE, elite_size=ELITE_SIZE, mut_rate=MUT_RATE):
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mut_rate = mut_rate
        self.neural_population = []
        self.rf_model = None
        self.best_neural_model = None
        self.training_history = []

    def train_rf(self, X_train, y_train, X_test, y_test):
        print("Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        rf_train_acc = self.rf_model.score(X_train, y_train)
        rf_test_acc = self.rf_model.score(X_test, y_test)
        print(f"RF - Train Accuracy: {rf_train_acc:.4f}, Test Accuracy: {rf_test_acc:.4f}")
        
        return rf_train_acc, rf_test_acc
    
    def train_neuroevolution(self, X_train, y_train, max_generations=MAX_GENERATIONS):
        print("\nTraining Neural Population with Neuroevolution...")
        self.neural_population = [Net() for _ in range(self.pop_size)]
        
        best_score = 0
        gen = 0
        stagnation_count = 0
        previous_best = 0
        
        while best_score < 0.85 or gen >= max_generations:
            gen += 1
            fitnesses = [evaluate_individual(m, X_train_t, y_train_t) for m in self.neural_population]
            ranked = sorted(zip(fitnesses, self.neural_population), key=lambda x: x[0], reverse=True)
            
            best_score = ranked[0][0]
            avg_score = np.mean(fitnesses)
            
            if abs(best_score - previous_best) < 0.001:
                stagnation_count += 1
            else:
                stagnation_count = 0
            previous_best = best_score
            
            self.training_history.append({
                'gen': gen,
                'best_fitness': best_score,
                'avg_fitness': avg_score,
                'stagnation_count': stagnation_count
            })
            
            print(f"Gen {gen:3d} | Best: {best_score:.4f} | Avg: {avg_score:.4f} | Stagnation: {stagnation_count}")
            
            if stagnation_count >= MAX_GENERATIONS/2:
                print("Early stopping due to stagnation")
                break
            
            elite = [m for _, m in ranked[:self.elite_size]]
            self.best_neural_model = elite[0]
            
            children = []
            while len(children) < POP_SIZE - ELITE_SIZE:
                p1, p2 = np.random.choice(elite, 2, replace=False)
                child = crossover(p1, p2)
                mutate(child)
                children.append(child)
                
                self.neural_population = elite + children
                
        print(f"Neuroevolution completed after {gen} generations")
        return best_score
    
    def evaluate_hybrid(self, X, y, top_n=10):
        rf_proba = self.rf_model.predict_proba(X)
        rf_pred = self.rf_model.predict(X)
        rf_acc = accuracy_score(y, rf_pred)
        
        neural_probas = []
        X_t = torch.tensor(X, dtype=torch.float32)
        
        for model in self.neural_population[:top_n]:
            with torch.no_grad():
                pred = F.softmax(model(X_t), dim=1)
                neural_probas.append(pred.numpy())
                
        avg_neural_proba = np.mean(neural_probas, axis=0)
        
        
        hybrid_proba = 0.5 * rf_proba + 0.5 * avg_neural_proba
        hybrid_pred = hybrid_proba.argmax(axis=1)
        hybrid_accuracy = accuracy_score(y, hybrid_pred)
        
        return {
            'rf_accuracy': rf_acc,
            'hybrid_accuracy': hybrid_accuracy,
            'improvement': hybrid_accuracy - rf_acc
        }
        
    def save_models(self, base_path="model"):
        os.makedirs(base_path, exist_ok=True)
        
        import joblib
        joblib.dump(self.rf_model, f"{base_path}/random_forest/random_forest.pkl")
        
        if self.best_neural_model:
            best_fitness = evaluate_individual(self.best_neural_model, X_train_t, y_train_t)
            fitness_str = f"{best_fitness:.4f}".replace('.', '_')
            torch.save({
                "model_state": self.best_neural_model.state_dict(),
                "fitness": best_fitness
            }, f"{base_path}/best_model_{fitness_str}.pt")
            
            joblib.dump(scaler, f"{base_path}/random_forest/scaler.pkl")
            
            metadata = {
                'input_size': input_size,
                'output_size': output_size,
                'hidden_size': HIDDEN_SIZE,
                'population_size': len(self.neural_population),
                'training_history': self.training_history
            }
            
            with open(f"{base_path}/training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
            print(f"Saved in '{base_path}' dir.")
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def main():
    print("Starting training...")
    
    trainer = HybridEnsembleTrainer()
    
    rf_train_acc, rf_test_acc = trainer.train_rf(X_train, y_train, X_test, y_test)

    best_neural_fitness = trainer.train_neuroevolution(X_train, y_train)
    
    print("\nEvaluating Hybrid Model...")
    train_results = trainer.evaluate_hybrid(X_train, y_train)
    test_results = trainer.evaluate_hybrid(X_test, y_test)
    
    print("\n=== FINAL RESULTS ===")
    print(f"Random Forest - Train: {rf_train_acc:.4f}, Test: {rf_test_acc:.4f}")
    print(f"Best Neural Model Fitness: {best_neural_fitness:.4f}")
    print(f"Hybrid Model - Train: {train_results['hybrid_accuracy']:.4f} "
          f"(Improvement: {train_results['improvement']:+.4f})")
    print(f"Hybrid Model - Test: {test_results['hybrid_accuracy']:.4f} "
          f"(Improvement: {test_results['improvement']:+.4f})")
    
    trainer.save_models()
    
    print("\nTraining completed! Models are ready for inference.")
    
    
if __name__ == "__main__":
    main()