"""ASTRA VISION — Phase 8F Taxonomy Guardrails and Reconciliation.

Enforces taxonomy protection rules between the supplied 5-class dataset
and the production 6-class classifier. Strictly prevents unauthorized
or naive automatic relabeling of ambiguous military categories.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from pydantic import BaseModel, Field

# Exact 6 production classes established in Phase 1-6
PRODUCTION_CLASSES: List[str] = [
    "Tank",
    "Military Vehicle",
    "Fighter Aircraft",
    "Helicopter",
    "Ship",
    "Drone",
]

# Exact 5 classes supplied in starter dataset
SUPPLIED_CLASSES: List[str] = [
    "aircraft",
    "drone",
    "helicopter",
    "military-vehicle",
    "naval",
]

# Known identified MBTs in the supplied military-vehicle category (from audit)
KNOWN_TANK_FILENAMES: Set[str] = {
    "images/military-vehicle/007-Object-184-3-T-72B3-RAE-2013-jpg.jpg",
    "images/military-vehicle/011-AlfredPalmerM3tank1942b-jpg.jpg",
    "images/military-vehicle/013-Crew-of-a-Sherman-tank-south-of-Vaucelles-jpg.jpg",
    "images/military-vehicle/014-Dutch-Panzerhaubitz-fires-in-Afghanistan-jpg.jpg",
    "images/military-vehicle/016-2014-Stepanakert-Czo-g-T-72-01-jpg.jpg",
    "images/military-vehicle/008-IDF-Magach-7-tank-jpg.jpg",
}

# Known fighter jets in the supplied aircraft category (from audit)
KNOWN_FIGHTER_FILENAMES: Set[str] = {
    "images/aircraft/001-F-22-Flaring-Cherry-Festival-jpg.jpg",
    "images/aircraft/002-GeoFS-F22-Raptor-png.png",
    "images/aircraft/003-GeoFS-FA18-Super-Hornet-png.png",
    "images/aircraft/004-HALTEDBFAeroindia2021-png.png",
    "images/aircraft/005-Havac-l-k-ve-Uzay-jpg.jpg",
    "images/aircraft/007-1981-Grumman-F-14A-Tomcat-jpg.jpg",
    "images/aircraft/010-Mikoyan-Gurevich-MiG-21-Fishbed-jpg.jpg",
    "images/aircraft/012-MiG-29K-carrier-borne-fighter-aircraft-jpg.jpg",
    "images/aircraft/016-Mirage-2000-display-jpg.jpg",
    "images/aircraft/020-Saab-JAS-39-Gripen-jpg.jpg",
    "images/aircraft/023-Sukhoi-Su-30MKI-jpg.jpg",
    "images/aircraft/027-Eurofighter-Typhoon-jpg.jpg",
    "images/aircraft/029-F-16-Fighting-Falcon-jpg.jpg",
}


class TaxonomyMapping(BaseModel):
    """Mapping descriptor between a supplied category and production taxonomy."""
    supplied_category: str
    target_production_class: Optional[str] = None
    is_ambiguous: bool
    direct_mapping_allowed: bool
    ambiguity_rationale: str
    potential_production_targets: List[str] = Field(default_factory=list)


TAXONOMY_RULES: Dict[str, TaxonomyMapping] = {
    "drone": TaxonomyMapping(
        supplied_category="drone",
        target_production_class="Drone",
        is_ambiguous=False,
        direct_mapping_allowed=True,
        ambiguity_rationale="Direct 1:1 semantic match with production 'Drone'.",
        potential_production_targets=["Drone"],
    ),
    "helicopter": TaxonomyMapping(
        supplied_category="helicopter",
        target_production_class="Helicopter",
        is_ambiguous=False,
        direct_mapping_allowed=True,
        ambiguity_rationale="Direct 1:1 semantic match with production 'Helicopter'.",
        potential_production_targets=["Helicopter"],
    ),
    "naval": TaxonomyMapping(
        supplied_category="naval",
        target_production_class="Ship",
        is_ambiguous=False,
        direct_mapping_allowed=True,
        ambiguity_rationale="Direct 1:1 semantic match with production 'Ship'.",
        potential_production_targets=["Ship"],
    ),
    "military-vehicle": TaxonomyMapping(
        supplied_category="military-vehicle",
        target_production_class=None,
        is_ambiguous=True,
        direct_mapping_allowed=False,
        ambiguity_rationale=(
            "Contains both Main Battle Tanks (MBTs) and general military vehicles (APCs, IFVs, trucks). "
            "In production, 'Tank' is an independent class distinct from 'Military Vehicle'. "
            "Automatic mapping is strictly prohibited to prevent model degradation."
        ),
        potential_production_targets=["Tank", "Military Vehicle"],
    ),
    "aircraft": TaxonomyMapping(
        supplied_category="aircraft",
        target_production_class=None,
        is_ambiguous=True,
        direct_mapping_allowed=False,
        ambiguity_rationale=(
            "Contains both combat/fighter jets and non-combat airframes (cargo, transports, bombers, trainers). "
            "In production, the class is specifically 'Fighter Aircraft'. "
            "Automatic mapping is strictly prohibited without human verification."
        ),
        potential_production_targets=["Fighter Aircraft"],
    ),
}


class FlaggedReviewItem(BaseModel):
    """An image requiring human taxonomy verification before any future training."""
    file_name: str
    supplied_category: str
    preliminary_hypothesis: str
    rationale: str
    needs_manual_review: bool = True


class TaxonomyAuditResult(BaseModel):
    """Result of auditing the dataset against taxonomy guardrails."""
    total_images: int
    unambiguous_count: int
    ambiguous_count: int
    direct_mappings: Dict[str, str]
    flagged_review_items: List[FlaggedReviewItem]
    guardrail_status: str = Field("ACTIVE_PROTECTION", description="Taxonomy protection status")
    recommendations: List[str]


def audit_taxonomy_alignment(labels_csv_path: Path) -> TaxonomyAuditResult:
    """Analyze labels.csv against taxonomy guardrails, flagging all ambiguous entries."""
    labels_csv_path = Path(labels_csv_path)
    if not labels_csv_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_csv_path}")

    unambiguous_count = 0
    ambiguous_count = 0
    flagged_items: List[FlaggedReviewItem] = []
    total_images = 0

    with open(labels_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_images += 1
            cat = row["category"]
            fn = row["file_name"]

            rule = TAXONOMY_RULES.get(cat)
            if not rule or rule.is_ambiguous:
                ambiguous_count += 1
                hypothesis = "Unknown"
                rationale = "Ambiguous category requires expert human review"

                if cat == "military-vehicle":
                    if fn in KNOWN_TANK_FILENAMES or any(k in fn.lower() for k in ["tank", "t-72", "sherman", "panzer"]):
                        hypothesis = "Tank"
                        rationale = "Identified as Main Battle Tank or tracked armor via visual/filename audit"
                    else:
                        hypothesis = "Military Vehicle"
                        rationale = "Wheeled transport, APC, IFV, or support vehicle"
                elif cat == "aircraft":
                    if fn in KNOWN_FIGHTER_FILENAMES or any(k in fn.lower() for k in ["f-22", "f22", "fa18", "fighter", "mig", "gripen", "sukhoi", "typhoon"]):
                        hypothesis = "Fighter Aircraft"
                        rationale = "Identified as high-performance combat/fighter aircraft"
                    else:
                        hypothesis = "Other Aircraft (Non-Fighter)"
                        rationale = "Transport, bomber, trainer, or general aviation"

                flagged_items.append(
                    FlaggedReviewItem(
                        file_name=fn,
                        supplied_category=cat,
                        preliminary_hypothesis=hypothesis,
                        rationale=rationale,
                        needs_manual_review=True,
                    )
                )
            else:
                unambiguous_count += 1

    return TaxonomyAuditResult(
        total_images=total_images,
        unambiguous_count=unambiguous_count,
        ambiguous_count=ambiguous_count,
        direct_mappings={
            "drone": "Drone",
            "helicopter": "Helicopter",
            "naval": "Ship",
        },
        flagged_review_items=flagged_items,
        guardrail_status="ACTIVE_PROTECTION",
        recommendations=[
            "Do NOT perform automatic relabeling during Phase 8E/8F.",
            "Retain supplied dataset files and categories strictly in their original state.",
            f"Review the {len(flagged_items)} flagged images in 'data/splits/taxonomy_reconciliation.json' before Phase 8G.",
            "Maintain held-out benchmark with its original 6 production classes.",
        ],
    )
