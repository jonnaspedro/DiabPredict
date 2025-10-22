import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# config/constants
TARGET_COL = "Outcome"
POP_SIZE = 75
ELITE_SIZE = 15
MUT_RATE = 0.10
HIDDEN_SIZE = 32    
MAX_GENS = 1000

df = pd.read_csv("data/diabetes.csv")
y = df[TARGET_COL].values
X = df.drop(columns=[TARGET_COL])

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
    
def evaluate(model):
    with torch.no_grad():
        pred = model(X_train_t)
        pred_labels = pred.argmax(dim=1)
        acc = (pred_labels == y_train_t).float().mean().item()
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
            
            
pop = [Net() for _ in range(POP_SIZE)]

best_score = 0
gen = 0
stagnation_count = 0
previous_best = 0

while best_score <= 0.85 and gen < MAX_GENS:
    gen += 1
    
    fitnesses = [evaluate(m) for m in pop]
    ranked = sorted(zip(fitnesses, pop), key=lambda x: x[0], reverse=True)
    
    best_score = ranked[0][0]
    avg_score = np.mean(fitnesses)
    
    if abs(best_score - previous_best) < 0.001:
        stagnation_count += 1
    else:
        stagnation_count = 0
    previous_best = best_score
    
    lines = (
        "|"
        f"{f'Gen {gen:04d}':^{12}}|"
        f"{f'Best: {best_score:.4f}':^{18}}|"
        f"{f'Average: {avg_score:.4f}':^{18}}|"
        f"{f'Stagnation: {stagnation_count:03d}':^{20}}"
        "|"
    )
    
    print(lines)
    print("-"*len(lines))
    
    if stagnation_count >= MAX_GENS/2:
        print("Early stopping due to stagnation")
        break
    
    elite = [m for _, m in ranked[:ELITE_SIZE]]
    
    children = []
    while len(children) < POP_SIZE - ELITE_SIZE:
        p1, p2 = np.random.choice(elite, 2, replace=False)
        child = crossover(p1, p2)
        mutate(child)
        children.append(child)
        
        pop = elite + children
        

best_model = max(pop, key=lambda m: evaluate(m))
best_fitness = f"{evaluate(best_model):.4f}".replace('.', '')
torch.save({
    "model_state": best_model.state_dict(),
    "scaler": scaler
}, f"model/best_model_f{best_fitness}_g{gen}_s{stagnation_count}.pt")

print(f"Best model saved: 'model/best_model_f{best_fitness}_g{gen}_s{stagnation_count}.pt'")