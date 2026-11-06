from pydantic import BaseModel, Field


class HeartDiseaseInput(BaseModel):
    age: int = Field(..., description="Age in years")
    sex: int = Field(..., description="Sex (1 = male; 0 = female)")
    cp: int = Field(..., description="Chest pain type (0-3)")
    trestbps: int = Field(..., description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    restecg: int = Field(..., description="Resting electrocardiographic results (0-2)")
    thalach: int = Field(..., description="Maximum heart rate achieved")
    exang: int = Field(..., description="Exercise-induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest")
    slope: int = Field(..., description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., description="Number of major vessels (0-4)")
    thal: int = Field(..., description="Thalassemia (dataset-encoded category)")
