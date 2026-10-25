"""
Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform
Module: Pydantic Validation Schemas & Data Models
File: backend/schemas.py

Defines request and response schemas for clinical patient data validation,
adverse reaction prediction, and dosage decision support.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# =====================================================================
# 1. Primary Clinical Patient Input Schema (Required by Specification)
# =====================================================================

class PatientData(BaseModel):
    """
    Patient Clinical Input Schema for mRNA Vaccine Reactogenicity Evaluation.
    Enforces strict typing and physiological value ranges.
    """
    age: int = Field(..., ge=18, le=100, description="Patient age in years (18 to 100)")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index in kg/m² (10.0 to 60.0)")
    has_allergies: int = Field(..., ge=0, le=1, description="History of allergies (0 = No, 1 = Yes)")
    prior_covid_infection: int = Field(..., ge=0, le=1, description="Prior documented SARS-CoV-2 infection (0 = No, 1 = Yes)")
    underlying_conditions: int = Field(..., ge=0, le=1, description="Presence of underlying chronic conditions (0 = No, 1 = Yes)")
    biomarker_wbc_count: float = Field(..., ge=1.0, le=30.0, description="White blood cell count in thousands/µL (1.0 to 30.0)")
    proposed_dosage_ml: float = Field(..., ge=0.1, le=1.0, description="Proposed vaccine dose in mL (0.10 to 1.00)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 45,
                "bmi": 27.4,
                "has_allergies": 1,
                "prior_covid_infection": 1,
                "underlying_conditions": 0,
                "biomarker_wbc_count": 8.5,
                "proposed_dosage_ml": 0.50
            }
        }


# =====================================================================
# 2. Primary Prediction Response Schema (Required by Specification)
# =====================================================================

class PredictionResponse(BaseModel):
    """
    Inference Response Schema returning adverse reaction risk and clinical guidance.
    """
    risk_probability: float = Field(
        ...,
        description="Adverse reaction risk probability rounded to 2 decimal places (0.00 to 1.00)",
        examples=[0.38]
    )
    risk_level: str = Field(
        ...,
        description="Categorical risk stratification: 'Low', 'Moderate', or 'High'",
        examples=["Moderate"]
    )
    recommendation: str = Field(
        ...,
        description="Clinical dosage guidance, monitoring protocol, or contraindication precautions",
        examples=[
            "Moderate reactogenicity risk identified (probability 0.38). Clinical monitoring advised; consider 30-minute observation post-vaccination."
        ]
    )


# =====================================================================
# 3. Extended Schemas for Platform Interoperability & Next.js Frontend
# =====================================================================

class DomainRiskDetail(BaseModel):
    score: float
    risk_tier: str
    clinical_indicators: List[str]

class ClinicalDomainRisks(BaseModel):
    cardiac: DomainRiskDetail
    anaphylaxis: DomainRiskDetail
    systemic: DomainRiskDetail
    hematologic: DomainRiskDetail

class FeatureImpact(BaseModel):
    feature: str
    label: str
    patient_value: Any
    reference_range: str
    status: str
    impact_score: float
    clinical_note: str

class ClinicalRecommendations(BaseModel):
    recommended_dosage_ug: float
    recommended_dosage_ml: float
    dosage_rationale: str
    observation_period_minutes: int
    contraindication_flag: bool
    contraindication_reason: Optional[str] = None
    pre_medication_required: bool
    pre_medication_notes: Optional[str] = None
    vital_signs_monitoring: List[str]

class DosageComparisonOption(BaseModel):
    dosage_ug: float
    dose_label: str
    predicted_risk_score: float
    risk_tier: str
    adverse_reaction_prob: float
    clinical_suitability: str
    is_recommended: bool

class AssessmentResponse(BaseModel):
    patient_id: str
    assessment_timestamp: str
    proposed_dosage_ug: float
    overall_risk_score: float
    risk_tier: str
    adverse_reaction_probability: float
    adverse_reaction_predicted: bool
    domain_risks: ClinicalDomainRisks
    top_contributing_factors: List[FeatureImpact]
    clinical_recommendations: ClinicalRecommendations
    dosage_comparisons: List[DosageComparisonOption]

class PatientClinicalInput(BaseModel):
    patient_id: Optional[str] = "MOD-PT-SIM"
    age: int = Field(default=35, ge=12, le=105)
    sex: str = Field(default="Male")
    bmi: float = Field(default=25.4, ge=12.0, le=65.0)
    hypertension: int = Field(default=0, ge=0, le=1)
    cardiovascular_disease: int = Field(default=0, ge=0, le=1)
    prior_myocarditis_pericarditis: int = Field(default=0, ge=0, le=1)
    diabetes: int = Field(default=0, ge=0, le=1)
    autoimmune_disease: int = Field(default=0, ge=0, le=1)
    history_severe_allergies: int = Field(default=0, ge=0, le=1)
    peg_polysorbate_allergy: int = Field(default=0, ge=0, le=1)
    immunocompromised: int = Field(default=0, ge=0, le=1)
    prior_covid_infection: int = Field(default=1, ge=0, le=1)
    months_since_last_infection: float = Field(default=6.0, ge=0.0, le=48.0)
    prior_vaccine_doses: int = Field(default=2, ge=0, le=6)
    days_since_last_dose: float = Field(default=180.0, ge=0.0, le=1200.0)
    troponin_i: float = Field(default=5.2, ge=0.1, le=250.0)
    crp: float = Field(default=1.8, ge=0.05, le=150.0)
    d_dimer: float = Field(default=0.28, ge=0.05, le=15.0)
    platelet_count: float = Field(default=245.0, ge=10.0, le=800.0)
    white_blood_cell_count: float = Field(default=6.8, ge=1.0, le=40.0)
    serum_ige: float = Field(default=42.0, ge=1.0, le=3000.0)
    proposed_dosage_ug: float = Field(default=50.0, ge=10.0, le=150.0)

class DosageSimulationRequest(BaseModel):
    patient_profile: PatientClinicalInput
    dosages_to_evaluate: Optional[List[float]] = None

class DosageSimulationResponse(BaseModel):
    patient_id: str
    simulations: List[DosageComparisonOption]
    optimal_dosage_ug: float
    optimization_summary: str

class BatchAssessmentRequest(BaseModel):
    patients: List[PatientClinicalInput]

class BatchAssessmentResponse(BaseModel):
    total_processed: int
    low_risk_count: int
    moderate_risk_count: int
    high_risk_count: int
    critical_risk_count: int
    high_priority_flags: List[Dict[str, Any]]
    summaries: List[Dict[str, Any]]

class SampleArchetype(BaseModel):
    id: str
    title: str
    subtitle: str
    badge: str
    description: str
    profile: PatientClinicalInput
