"""
Pydantic request/response schemas for the ChurnOps API.

WHAT: Pydantic validates incoming JSON against these type definitions
      before any of our code runs.
WHY:  Without validation, a malformed request (e.g. tenure as a string,
      or a missing field) would fail deep inside the model pipeline
      with a confusing error. Pydantic catches it at the door with a
      clear 422 error instead.
WHERE: Used by app/main.py to type the /predict request body and
       response.
"""

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    tenure: int = Field(ge=0, description="Number of months the customer has stayed")
    monthly_charges: float = Field(ge=0, description="Current monthly bill amount")
    total_charges: float = Field(ge=0, description="Total amount billed to the customer so far")
    contract: str = Field(description="e.g. 'Month-to-month', 'One year', 'Two year'")
    internet_service: str = Field(description="e.g. 'DSL', 'Fiber optic', 'No'")
    senior_citizen: int = Field(ge=0, le=1, description="1 if senior citizen, else 0")
    partner: int = Field(ge=0, le=1, description="1 if the customer has a partner, else 0")
    dependents: int = Field(ge=0, le=1, description="1 if the customer has dependents, else 0")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 12,
                "monthly_charges": 75.5,
                "total_charges": 850.0,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "senior_citizen": 0,
                "partner": 1,
                "dependents": 0,
            }
        }
    }


class ChurnResponse(BaseModel):
    # protected_namespaces=() silences Pydantic's warning about fields
    # starting with "model_" - here that prefix is just our naming
    # convention, not a conflict with anything Pydantic itself defines.
    model_config = {"protected_namespaces": ()}

    churn: bool
    probability: float
    model_version: str


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    model_version: str
    model_alias: str


class HealthResponse(BaseModel):
    status: str
