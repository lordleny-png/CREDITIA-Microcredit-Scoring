
"""
CREDTIA - Credit Scoring Microcredit Togo
Version Python prototype based on the CREDTIA V1 Excel model.

Architecture:
1) Mandatory controls / stop rules
2) Capacity calculation by income profile
3) Expert score /100
4) Data-quality score /100
5) Optional XGBoost model trained on observed repayment outcomes
6) Human validation remains mandatory

IMPORTANT:
- CREDTIA V1 is an expert model; its A-E classes are NOT probabilities of default.
- XGBoost requires historical applications with an observed target such as default_90d.
- Mandatory stop rules must never be overridden by a high ML score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

# Optional ML imports are kept here so the rule engine can still be used
# if xgboost has not yet been installed.
try:
    from xgboost import XGBClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, recall_score,
        precision_score, f1_score, confusion_matrix
    )
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ============================================================
# 1. CREDTIA V1 PARAMETERS
# ============================================================

PARAMS = {
    "effort_standard": 0.30,
    "effort_exception": 0.35,
    "salary_quotity": 0.35,       # provisional; compliance must validate
    "dscr": 1.25,
    "independent_stress": 0.80,
    "seasonal_stress": 0.70,
    "rounding": 5000,

    # Cost-of-living / minimum remaining income assumptions
    "rav": {
        "Lomé": (35000, 10000),
        "Autre urbain": (30000, 8000),
        "Rural": (25000, 7000),
    },

    # Capacity utilization -> quantitative points /25
    "capacity_points": [
        (0.70, 25),
        (0.85, 20),
        (1.00, 15),
        (np.inf, 0),
    ],

    # Volatility -> points /10
    "volatility_points": [
        (0.10, 10),
        (0.20, 8),
        (0.35, 5),
        (np.inf, 2),
    ],

    # Existing debt / reference flow -> points /10
    "debt_points": [
        (0.00, 10),
        (0.10, 8),
        (0.20, 5),
        (np.inf, 2),
    ],

    # Savings / proposed installment -> points /5
    "savings_points": [
        (3.00, 5),
        (2.00, 4),
        (1.00, 3),
        (0.01, 1),
        (0.00, 0),
    ],

    # Risk classes
    "risk_classes": {
        "A": 80,
        "B": 70,
        "C": 60,
        "D": 50,
        "E": 0,
    },

    # Data quality thresholds
    "quality_satisfactory": 80,
    "quality_review": 60,
}


QUALITATIVE_WEIGHTS = {
    "continuite_emploi_activite": 8,
    "coherence_credit": 8,
    "discipline_gestion": 7,
    "resilience_diversification": 7,
    "stabilite_employeur_marche": 4,
    "stabilite_contact": 3,
    "garantie_soutien": 3,
}


# ============================================================
# 2. DATA STRUCTURE
# ============================================================

@dataclass
class Application:
    dossier_id: str
    profile: str  # "salarie", "independant", "saisonnier"
    requested_amount: float
    term_periods: int
    annual_rate: float
    household_size: int
    cost_zone: str

    existing_debt_payment: float = 0.0
    liquid_savings: float = 0.0
    proposed_installment: float = 0.0

    # Common eligibility
    kyc_complete: bool = True
    bic_consent: bool = True
    bic_result: str = "Aucun engagement / à jour"
    internal_history: str = "Sans incident / échéances respectées"
    active_arrears_over_30d: bool = False
    fraud_or_major_inconsistency: bool = False
    authorized_purpose: bool = True
    age_term_compliant: bool = True

    # Employee / pensioner
    net_monthly_income: float = 0.0
    essential_household_charges_monthly: float = 0.0
    income_volatility: float = 0.0
    proof_months: int = 0
    salary_quotity_confirmed: float = 0.0
    capacity_mode: str = "Standard"

    # Independent / merchant
    verified_collections: float = 0.0
    variable_costs: float = 0.0
    fixed_business_charges: float = 0.0
    household_charges: float = 0.0
    cashflow_volatility: float = 0.0
    observed_months: int = 0

    # Seasonal / mixed
    cycle_collections: float = 0.0
    cycle_production_costs: float = 0.0
    cycle_household_charges: float = 0.0
    cycle_length_months: int = 1
    cycle_volatility: float = 0.0
    observed_cycles: int = 0

    # Seven qualitative ratings: 0 to 4
    qualitative: Optional[Dict[str, int]] = None

    # Data-quality ratings: 0 to 4
    proof_quality: int = 3
    consistency_quality: int = 3
    freshness_quality: int = 3


# ============================================================
# 3. CORE CALCULATIONS
# ============================================================

def annuity_payment(principal: float, annual_rate: float,
                    periods: int, period_months: int = 1) -> float:
    """Monthly or periodic annuity payment."""
    if principal <= 0 or periods <= 0:
        return 0.0
    r = annual_rate * period_months / 12
    if r == 0:
        return principal / periods
    return principal * r / (1 - (1 + r) ** (-periods))


def remaining_living_amount(app: Application) -> float:
    base, extra = PARAMS["rav"].get(app.cost_zone, PARAMS["rav"]["Rural"])
    monthly = base + max(0, app.household_size - 1) * extra
    period_months = app.cycle_length_months if app.profile == "saisonnier" else 1
    return monthly * period_months


def capacity_max(app: Application) -> float:
    """Maximum sustainable periodic installment according to CREDTIA V1."""
    rav = remaining_living_amount(app)

    if app.profile == "salarie":
        effort = (
            PARAMS["effort_exception"]
            if app.capacity_mode.lower().startswith("exception")
            else PARAMS["effort_standard"]
        )
        quotity = (
            app.salary_quotity_confirmed
            if app.salary_quotity_confirmed > 0
            else PARAMS["salary_quotity"]
        )

        c1 = effort * app.net_monthly_income - app.existing_debt_payment
        c2 = quotity * app.net_monthly_income - app.existing_debt_payment
        c3 = (
            app.net_monthly_income
            - app.essential_household_charges_monthly
            - app.existing_debt_payment
            - rav
        )
        return max(0.0, min(c1, c2, c3))

    if app.profile == "independant":
        prudent = app.verified_collections * PARAMS["independent_stress"]
        surplus = (
            prudent
            - app.variable_costs
            - app.fixed_business_charges
            - app.household_charges
            - rav
        )
        before_existing_debt = surplus / PARAMS["dscr"]
        return max(0.0, before_existing_debt - app.existing_debt_payment)

    if app.profile == "saisonnier":
        prudent = app.cycle_collections * PARAMS["seasonal_stress"]
        surplus = (
            prudent
            - app.cycle_production_costs
            - app.cycle_household_charges
            - rav
        )
        return max(0.0, surplus / PARAMS["dscr"] - app.existing_debt_payment)

    raise ValueError("profile must be salarie, independant or saisonnier")


def max_sustainable_amount(app: Application, max_installment: float) -> float:
    period_months = app.cycle_length_months if app.profile == "saisonnier" else 1
    r = app.annual_rate * period_months / 12

    if max_installment <= 0 or app.term_periods <= 0:
        return 0.0
    if r == 0:
        return max_installment * app.term_periods

    return max_installment * (1 - (1 + r) ** (-app.term_periods)) / r


def recommended_amount(app: Application, max_amount: float) -> float:
    raw = min(app.requested_amount, max_amount)
    step = PARAMS["rounding"]
    if step <= 0:
        return max(0.0, raw)
    return max(0.0, np.floor(raw / step) * step)


# ============================================================
# 4. MANDATORY STOP RULES
# ============================================================

def mandatory_stops(app: Application, capacity: float) -> Dict[str, Any]:
    stops = {}

    stops["kyc"] = not app.kyc_complete
    stops["bic"] = not (app.bic_consent and app.bic_result != "Non consulté")
    stops["fraud"] = app.fraud_or_major_inconsistency
    stops["purpose"] = not app.authorized_purpose
    stops["age_term"] = not app.age_term_compliant
    stops["active_arrears"] = app.active_arrears_over_30d or (
        app.bic_result == "Impayé grave actif"
    )
    stops["capacity"] = capacity <= 0

    return {
        "stops": stops,
        "number_of_stops": int(sum(stops.values())),
        "blocked": any(stops.values()),
    }


# ============================================================
# 5. EXPERT SCORE /100
# ============================================================

def bracket(value: float, brackets) -> float:
    for limit, points in brackets:
        if value <= limit:
            return points
    return 0.0


def expert_quantitative_score(app: Application, installment: float,
                              capacity: float) -> Dict[str, float]:
    utilization = 9.99 if capacity <= 0 else installment / capacity

    if app.profile == "salarie":
        reference_flow = max(0.0, app.net_monthly_income)
        volatility = app.income_volatility
    elif app.profile == "independant":
        reference_flow = max(
            0.0, app.verified_collections
            - app.variable_costs
            - app.fixed_business_charges
        )
        volatility = app.cashflow_volatility
    else:
        reference_flow = max(
            0.0, app.cycle_collections - app.cycle_production_costs
        )
        volatility = app.cycle_volatility

    debt_ratio = (
        9.99 if reference_flow <= 0
        else app.existing_debt_payment / reference_flow
    )

    capacity_points = bracket(utilization, PARAMS["capacity_points"])
    volatility_points = bracket(volatility, PARAMS["volatility_points"])
    debt_points = bracket(debt_ratio, PARAMS["debt_points"])

    history_internal = {
        "Sans incident / échéances respectées": 5,
        "Retards ≤ 7 jours, régularisés": 4,
        "Retards de 8 à 30 jours, régularisés": 2,
        "Aucun historique": 3,
    }.get(app.internal_history, 0)

    bic_points = {
        "Aucun engagement / à jour": 5,
        "Historique limité": 3,
        "Retards mineurs clos": 2,
    }.get(app.bic_result, 0)

    history_points = history_internal + bic_points

    savings_ratio = (
        0 if installment <= 0
        else app.liquid_savings / installment
    )
    savings_points = bracket(savings_ratio, PARAMS["savings_points"])

    total = (
        capacity_points
        + volatility_points
        + debt_points
        + history_points
        + savings_points
    )

    return {
        "capacity_points": capacity_points,
        "volatility_points": volatility_points,
        "debt_points": debt_points,
        "history_bic_points": history_points,
        "savings_points": savings_points,
        "quantitative_total": total,
        "capacity_utilization": utilization,
        "debt_ratio": debt_ratio,
    }


def expert_qualitative_score(app: Application) -> Dict[str, float]:
    q = app.qualitative or {k: 0 for k in QUALITATIVE_WEIGHTS}

    points = {}
    for key, weight in QUALITATIVE_WEIGHTS.items():
        level = int(np.clip(q.get(key, 0), 0, 4))
        points[key] = level / 4 * weight

    points["qualitative_total"] = sum(points.values())
    return points


# ============================================================
# 6. DATA QUALITY /100
# ============================================================

def data_quality_score(app: Application) -> Dict[str, float]:
    # Essential fields are checked by profile.
    common_values = [
        app.dossier_id, app.requested_amount, app.term_periods,
        app.annual_rate, app.household_size, app.cost_zone,
        app.existing_debt_payment, app.kyc_complete,
        app.bic_consent, app.bic_result, app.authorized_purpose,
        app.age_term_compliant,
    ]
    common_complete = sum(v is not None and v != "" for v in common_values)

    if app.profile == "salarie":
        module_values = [
            app.net_monthly_income,
            app.essential_household_charges_monthly,
            app.income_volatility,
            app.proof_months,
        ]
    elif app.profile == "independant":
        module_values = [
            app.verified_collections,
            app.variable_costs,
            app.fixed_business_charges,
            app.household_charges,
            app.cashflow_volatility,
            app.observed_months,
        ]
    else:
        module_values = [
            app.cycle_collections,
            app.cycle_production_costs,
            app.cycle_household_charges,
            app.cycle_length_months,
            app.cycle_volatility,
            app.observed_cycles,
        ]

    q = app.qualitative or {}
    qualitative_complete = sum(
        k in q and q[k] is not None for k in QUALITATIVE_WEIGHTS
    )

    # Same principle as the Excel: completeness is separate from risk.
    completeness = 35 * (
        (common_complete + sum(v is not None for v in module_values)
         + qualitative_complete)
        / (len(common_values) + len(module_values) + len(QUALITATIVE_WEIGHTS))
    )

    proof = app.proof_quality / 4 * 30
    consistency = app.consistency_quality / 4 * 20
    freshness = app.freshness_quality / 4 * 15

    total = completeness + proof + consistency + freshness

    return {
        "completeness": completeness,
        "proof": proof,
        "consistency": consistency,
        "freshness": freshness,
        "quality_total": total,
    }


# ============================================================
# 7. FULL CREDTIA ENGINE
# ============================================================

def score_application(app: Application) -> Dict[str, Any]:
    capacity = capacity_max(app)

    period_months = app.cycle_length_months if app.profile == "saisonnier" else 1

    calculated_installment = annuity_payment(
        app.requested_amount,
        app.annual_rate,
        app.term_periods,
        period_months
    )

    installment = (
        app.proposed_installment
        if app.proposed_installment > 0
        else calculated_installment
    )

    max_amount = max_sustainable_amount(app, capacity)
    rec_amount = recommended_amount(app, max_amount)

    stops = mandatory_stops(app, capacity)
    quant = expert_quantitative_score(app, installment, capacity)
    qual = expert_qualitative_score(app)
    quality = data_quality_score(app)

    risk_score = quant["quantitative_total"] + qual["qualitative_total"]

    if risk_score >= 80:
        risk_class = "A"
    elif risk_score >= 70:
        risk_class = "B"
    elif risk_score >= 60:
        risk_class = "C"
    elif risk_score >= 50:
        risk_class = "D"
    else:
        risk_class = "E"

    if stops["blocked"]:
        recommendation = "BLOQUÉ — validation humaine requise"
    elif quality["quality_total"] < 60:
        recommendation = "DOSSIER À COMPLÉTER"
    elif installment > capacity:
        recommendation = "RÉVISER MONTANT / DURÉE"
    elif quality["quality_total"] < 80:
        recommendation = "REVUE HUMAINE RENFORCÉE"
    elif risk_class in ("A", "B"):
        recommendation = "AVIS FAVORABLE — validation humaine"
    elif risk_class == "C":
        recommendation = "REVUE HUMAINE / CONDITIONS"
    else:
        recommendation = "AVIS DÉFAVORABLE — revue humaine"

    return {
        "dossier_id": app.dossier_id,
        "profile": app.profile,
        "calculated_installment": calculated_installment,
        "installment_used": installment,
        "max_installment": capacity,
        "max_sustainable_amount": max_amount,
        "recommended_amount": rec_amount,
        "risk_score": risk_score,
        "risk_class": risk_class,
        "data_quality": quality["quality_total"],
        "recommendation": recommendation,
        **quant,
        **qual,
        **quality,
        **stops,
    }


# ============================================================
# 8. XGBOOST: TRAIN ON REAL REPAYMENT HISTORY
# ============================================================

# Do NOT include direct sensitive attributes or direct geographic identifiers.
# Zone can influence the expert remaining-living calculation, but is not a
# direct ML predictor here.
ML_FEATURES = [
    "profile",
    "requested_amount",
    "term_periods",
    "annual_rate",
    "household_size",
    "existing_debt_payment",
    "liquid_savings",
    "net_monthly_income",
    "essential_household_charges_monthly",
    "income_volatility",
    "proof_months",
    "verified_collections",
    "variable_costs",
    "fixed_business_charges",
    "household_charges",
    "cashflow_volatility",
    "observed_months",
    "cycle_collections",
    "cycle_production_costs",
    "cycle_household_charges",
    "cycle_length_months",
    "cycle_volatility",
    "observed_cycles",
    "expert_score",
    "data_quality",
    "capacity_utilization",
    "debt_ratio",
]


def applications_to_dataframe(applications):
    rows = []
    for app in applications:
        result = score_application(app)
        row = {f: getattr(app, f, np.nan) for f in ML_FEATURES if hasattr(app, f)}
        row["expert_score"] = result["risk_score"]
        row["data_quality"] = result["data_quality"]
        row["capacity_utilization"] = result["capacity_utilization"]
        row["debt_ratio"] = result["debt_ratio"]
        rows.append(row)
    return pd.DataFrame(rows)


def train_xgboost(
    df: pd.DataFrame,
    target: str = "default_90d",
    test_size: float = 0.25,
    random_state: int = 42,
):
    """
    Train XGBoost on observed historical repayment outcomes.

    target:
        1 = observed default according to the institution's approved definition
        0 = non-default

    The target definition must be fixed before training.
    """
    if not XGB_AVAILABLE:
        raise ImportError(
            "Install dependencies first: pip install xgboost scikit-learn pandas numpy"
        )

    if target not in df.columns:
        raise ValueError(
            f"Missing target '{target}'. CREDTIA needs historical repayment outcomes."
        )

    features = [c for c in ML_FEATURES if c in df.columns]
    X = df[features].copy()
    y = df[target].astype(int)

    if y.nunique() < 2:
        raise ValueError("The target must contain both 0 and 1.")

    categorical = [c for c in features if X[c].dtype == "object"]
    numeric = [c for c in features if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    positives = max(1, int(y_train.sum()))
    negatives = max(1, int((y_train == 0).sum()))
    scale_pos_weight = negatives / positives

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.50).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "default_rate": float(y.mean()),
    }

    return pipeline, metrics, features


def predict_default_probability(pipeline, app: Application) -> float:
    """Return XGBoost estimated PD only after the model has been trained."""
    df = applications_to_dataframe([app])
    return float(pipeline.predict_proba(df)[:, 1][0])


# ============================================================
# 9. HYBRID OUTPUT
# ============================================================

def hybrid_assessment(app: Application, xgb_pipeline=None) -> Dict[str, Any]:
    """
    CREDTIA hybrid architecture.

    The expert engine remains responsible for mandatory controls,
    capacity and transparent score. XGBoost adds a statistical risk
    signal when a validated historical model exists.
    """
    result = score_application(app)

    if xgb_pipeline is not None:
        result["xgboost_pd"] = predict_default_probability(xgb_pipeline, app)
        result["ml_available"] = True
    else:
        result["xgboost_pd"] = None
        result["ml_available"] = False

    # The ML signal is deliberately not allowed to override a mandatory stop.
    result["human_validation_required"] = True

    return result


if __name__ == "__main__":
    # Example close to the Excel demonstration.
    example = Application(
        dossier_id="EXEMPLE-001",
        profile="salarie",
        requested_amount=500000,
        term_periods=12,
        annual_rate=0.20,
        household_size=3,
        cost_zone="Lomé",
        existing_debt_payment=25000,
        liquid_savings=150000,
        net_monthly_income=300000,
        essential_household_charges_monthly=120000,
        income_volatility=0.05,
        proof_months=6,
        qualitative={
            "continuite_emploi_activite": 4,
            "coherence_credit": 3,
            "discipline_gestion": 3,
            "resilience_diversification": 2,
            "stabilite_employeur_marche": 3,
            "stabilite_contact": 4,
            "garantie_soutien": 2,
        },
        proof_quality=3,
        consistency_quality=3,
        freshness_quality=3,
    )

    result = score_application(example)
    for k, v in result.items():
        print(f"{k}: {v}")
