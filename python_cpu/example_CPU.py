import time
import numpy as np
import os
import multiprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

print("="*60)
print("Random Forest Grid Search - HPC Demo")
print("="*60)

# Detect number of CPUs
num_cpus = multiprocessing.cpu_count()
# Also check SLURM allocation if available
slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', None)

print(f"\n[INFO] CPU Detection:")
print(f"  - Total CPUs available: {num_cpus}")
if slurm_cpus:
    print(f"  - SLURM allocated CPUs: {slurm_cpus}")
    print(f"  - Using SLURM allocation")
else:
    print(f"  - No SLURM allocation detected")
    print(f"  - Using all available CPUs")

# Generate synthetic dataset
print(f"\n[STEP 1/4] Generating synthetic dataset...")
print(f"  - Creating 10,000 samples with 20 features")
start_gen = time.time()
X, y = make_classification(
    n_samples=50000, n_features=20,
    n_informative=10, n_redundant=5, random_state=42
)
gen_time = time.time() - start_gen
print(f"  - Dataset shape: {X.shape}")
print(f"  - Generation time: {gen_time:.2f} seconds")

# Define hyperparameter grid
param_grid = {
    'n_estimators': [500, 1000],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

print(f"\n[STEP 2/4] Setting up Grid Search...")
print(f"  - Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
print(f"  - Cross-validation folds: 5")
print(f"  - Total fits: {np.prod([len(v) for v in param_grid.values()]) * 5}")
print(f"  - Parallelization: n_jobs=-1 (all CPUs)")

print(f"\n[STEP 3/4] Starting Grid Search...")
print(f"  - This may take several minutes depending on CPU allocation")
print(f"  - Progress will be shown below:")
print("-"*60)

start_time = time.time()

# Grid search with parallel execution
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=5,
    n_jobs=-1,  # Use all available cores!
    verbose=2
)

# Fit the model
grid_search.fit(X, y)

elapsed_time = time.time() - start_time

print("-"*60)
print(f"\n[STEP 4/4] Training Results:")
print(f"  - Total training time: {elapsed_time/60:.2f} minutes ({elapsed_time:.1f} seconds)")
print(f"  - Average time per parameter combination: {elapsed_time / np.prod([len(v) for v in param_grid.values()]):.1f} seconds")
print(f"\n[RESULTS] Best Model Performance:")
print(f"  - Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"  - Best parameters:")
for param, value in grid_search.best_params_.items():
    print(f"      {param}: {value}")

# Calculate efficiency
print(f"\n[PERFORMANCE METRICS]")
print(f"  - Number of CPUs used: {num_cpus}")
if slurm_cpus:
    print(f"  - SLURM allocation: {slurm_cpus} CPUs")
print(f"  - Wall-clock time: {elapsed_time:.1f} seconds")
print(f"  - Estimated CPU-hours: {(elapsed_time * num_cpus) / 3600:.2f} hours")

# Save the best model
#print(f"\n[SAVING] Saving best model...")
#joblib.dump(grid_search.best_estimator_, 'best_model.pkl')
#print(f"  - Model saved to: best_model.pkl")

# Save detailed results
print(f"[SAVING] Saving results to file...")
with open('results.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("Random Forest Grid Search Results\n")
    f.write("="*60 + "\n\n")
    f.write(f"Training time: {elapsed_time/60:.2f} minutes ({elapsed_time:.1f} seconds)\n")
    f.write(f"Number of CPUs available: {num_cpus}\n")
    if slurm_cpus:
        f.write(f"SLURM allocated CPUs: {slurm_cpus}\n")
    f.write(f"\nBest cross-validation score: {grid_search.best_score_:.4f}\n")
    f.write(f"\nBest parameters:\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"  - {param}: {value}\n")
    f.write(f"\nDataset information:\n")
    f.write(f"  - Samples: {X.shape[0]}\n")
    f.write(f"  - Features: {X.shape[1]}\n")
    f.write(f"\nGrid search settings:\n")
    f.write(f"  - Parameter combinations tested: {np.prod([len(v) for v in param_grid.values()])}\n")
    f.write(f"  - Cross-validation folds: 5\n")
    f.write(f"  - Total model fits: {np.prod([len(v) for v in param_grid.values()]) * 5}\n")
    f.write(f"  - Parallelization: n_jobs=-1 (all available CPUs)\n")
    f.write(f"\nPerformance metrics:\n")
    f.write(f"  - Wall-clock time: {elapsed_time:.1f} seconds\n")
    f.write(f"  - Estimated CPU-hours: {(elapsed_time * num_cpus) / 3600:.2f} hours\n")
    f.write(f"  - Average time per parameter combo: {elapsed_time / np.prod([len(v) for v in param_grid.values()]):.1f} seconds\n")

print(f"  - Results saved to: results.txt")

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)