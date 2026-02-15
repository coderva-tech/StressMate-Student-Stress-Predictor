import pandas as pd
import numpy as np

# Number of samples
n_samples = 10000

# Seed for reproducibility
np.random.seed(42)

# 1. Generate features
sleep_quality = np.random.randint(1, 6, n_samples)        # 1–5
headache_freq = np.random.randint(1, 8, n_samples)        # 1–7 times/week
academic_performance = np.random.randint(1, 6, n_samples) # 1–5
study_load = np.random.randint(1, 6, n_samples)           # 1–5
extracurricular = np.random.randint(1, 6, n_samples)      # 1–5

# 2. Generate stress_level based on simple rules
# Base stress = study_load + headache_freq - sleep_quality - extracurricular
stress_level = study_load + headache_freq - sleep_quality - extracurricular

# Clip stress_level to 1–5
stress_level = np.clip(stress_level, 1, 5)

# 3. Create DataFrame
df = pd.DataFrame({
    'sleep_quality': sleep_quality,
    'headache_freq': headache_freq,
    'academic_performance': academic_performance,
    'study_load': study_load,
    'extracurricular': extracurricular,
    'stress_level': stress_level
})

# 4. Save to CSV
df.to_csv('synthetic_student_stress.csv', index=False)

print("Synthetic dataset generated with", n_samples, "samples.")
