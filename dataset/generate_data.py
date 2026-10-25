"""
Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform
Module: Synthetic Patient Clinical Dataset Generator
File: dataset/generate_data.py

Generates 1,500 synthetic patient records using pandas and numpy, incorporating
key patient demographics, clinical history, laboratory biomarkers (WBC count),
and mRNA vaccine dosage volume to calculate the adverse reaction risk label.
"""

import os
import numpy as np
import pandas as pd


def generate_patient_safety_dataset(
    n_samples: int = 1500,
    random_seed: int = 42,
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate synthetic clinical patient records for mRNA vaccine reactogenicity assessment.

    Parameters:
        n_samples (int): Number of synthetic patient records to generate (default: 1500).
        random_seed (int): Seed for reproducibility (default: 42).
        output_path (str, optional): Target CSV filepath. Defaults to 'dataset/patient_safety_data.csv'.

    Returns:
        pd.DataFrame: Generated dataset containing patient features and target risk label.
    """
    np.random.seed(random_seed)

    # 1. Demographic & Physiological Features
    # Age: integer between 18 and 85 inclusive
    age = np.random.randint(18, 86, size=n_samples)

    # BMI: float between 18.5 and 40.0 (rounded to 1 decimal place)
    bmi = np.round(np.random.uniform(18.5, 40.0, size=n_samples), 1)

    # 2. Clinical History & Comorbidities (Binary: 0 or 1)
    # has_allergies: ~24% clinical prevalence (food/drug/environmental atopy)
    has_allergies = np.random.binomial(1, p=0.24, size=n_samples)

    # prior_covid_infection: ~60% population seropositivity / documented infection
    prior_covid_infection = np.random.binomial(1, p=0.60, size=n_samples)

    # underlying_conditions: ~32% prevalence (cardiovascular, diabetes, hypertension, autoimmune)
    underlying_conditions = np.random.binomial(1, p=0.32, size=n_samples)

    # 3. Laboratory Biomarkers
    # biomarker_wbc_count: float between 4.0 and 15.0 thousands per microliter (rounded to 2 decimals)
    # Normal reference range: 4.5 - 11.0; elevated values indicate systemic inflammatory priming
    biomarker_wbc_count = np.round(np.random.uniform(4.0, 15.0, size=n_samples), 2)

    # 4. Proposed Vaccine Dosage Volume
    # proposed_dosage_ml: float between 0.25 mL and 0.50 mL (rounded to 2 decimals)
    # 0.25 mL (adapted/pediatric dose, e.g. 25-50 µg mRNA) up to 0.50 mL (standard primary 100 µg mRNA)
    proposed_dosage_ml = np.round(np.random.uniform(0.25, 0.50, size=n_samples), 2)

    # 5. Target Label Calculation (adverse_reaction_risk)
    # Pathophysiological rule combining:
    # - Elevated WBC count (> 10.5 k/µL) as a marker of heightened immune/inflammatory priming
    # - Pre-existing underlying conditions
    # - Allergic diathesis (hypersensitivity susceptibility)
    # - Higher mRNA dosage volume (0.50 mL delivers higher lipid nanoparticle & mRNA mass)
    # - Specific high-risk interactions (e.g., high dosage combined with high WBC count and conditions)

    # Normalized dosage index between 0.0 (0.25 mL) and 1.0 (0.50 mL)
    dosage_norm = (proposed_dosage_ml - 0.25) / 0.25

    # Composite risk score calculation
    risk_score = (
        0.30 * (biomarker_wbc_count > 10.5).astype(float) +
        0.25 * underlying_conditions.astype(float) +
        0.20 * has_allergies.astype(float) +
        0.25 * dosage_norm +
        0.15 * ((biomarker_wbc_count > 10.0) & (proposed_dosage_ml >= 0.40)).astype(float) +
        0.10 * ((underlying_conditions == 1) & (proposed_dosage_ml >= 0.40)).astype(float) +
        0.08 * ((has_allergies == 1) & (proposed_dosage_ml >= 0.45)).astype(float) +
        0.05 * (bmi > 32.0).astype(float) +
        np.random.normal(0.0, 0.05, size=n_samples)  # Realistic clinical biological variance
    )

    # Threshold chosen to yield ~30% High Risk (1) and ~70% Safe (0) cohort distribution
    adverse_reaction_risk = (risk_score >= 0.55).astype(int)

    # Construct DataFrame
    df = pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "has_allergies": has_allergies,
        "prior_covid_infection": prior_covid_infection,
        "underlying_conditions": underlying_conditions,
        "biomarker_wbc_count": biomarker_wbc_count,
        "proposed_dosage_ml": proposed_dosage_ml,
        "adverse_reaction_risk": adverse_reaction_risk
    })

    # Determine default destination if not provided
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "patient_safety_data.csv")

    # Ensure parent directory exists and write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} synthetic patient records.")
    print(f"File saved to: {output_path}")
    print(f"Target Label Breakdown ('adverse_reaction_risk'):")
    print(df["adverse_reaction_risk"].value_counts(normalize=True).rename(
        {0: "0 (Safe)", 1: "1 (High Risk)"}
    ).to_string())

    return df


if __name__ == "__main__":
    generate_patient_safety_dataset()
