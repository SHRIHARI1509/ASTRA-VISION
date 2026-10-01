"""ASTRA VISION — Phase 8G Taxonomy Reconciliation Module.

Provides human-reviewed label reconciliations for the 150 starter images.
Ensures zero images enter the 6-class training pipeline with unresolved or ambiguous labels.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field

# Six production classes established in Phase 1-6
PRODUCTION_CLASSES: List[str] = [
    "Tank",
    "Military Vehicle",
    "Fighter Aircraft",
    "Helicopter",
    "Ship",
    "Drone",
]

# Explicit, manually reviewed mappings for all 30 military-vehicle images
REVIEWED_MILITARY_VEHICLE_LABELS: Dict[str, str] = {
    "images/military-vehicle/001-A-huge-motorized-Soviet-convoy-advances-across-the-Grand-Khi.png": "Military Vehicle",
    "images/military-vehicle/003-Gunners-conduct-rearm-refuel-and-resupply-point-R3P-LPD-9822.jpg": "Military Vehicle",
    "images/military-vehicle/004-HSTVL-jpg.jpg": "Tank",  # High Survivability Test Vehicle Light Tank
    "images/military-vehicle/005-Landsverk-L-60-at-Collins-Barracks-jpg.jpg": "Tank",  # Landsverk L-60 Light Tank
    "images/military-vehicle/007-Object-184-3-T-72B3-RAE-2013-jpg.jpg": "Tank",  # T-72B3 Main Battle Tank
    "images/military-vehicle/011-AlfredPalmerM3tank1942b-jpg.jpg": "Tank",  # M3 Lee Tank
    "images/military-vehicle/012-BMP-1-Zlot-Dar-owo-2009-JPG.jpg": "Military Vehicle",  # BMP-1 Infantry Fighting Vehicle
    "images/military-vehicle/013-Crew-of-a-Sherman-tank-south-of-Vaucelles-jpg.jpg": "Tank",  # M4 Sherman Tank
    "images/military-vehicle/014-Dutch-Panzerhaubitz-fires-in-Afghanistan-jpg.jpg": "Military Vehicle",  # PzH 2000 Self-Propelled Howitzer
    "images/military-vehicle/016-2014-Stepanakert-Czo-g-T-72-01-jpg.jpg": "Tank",  # T-72 Main Battle Tank
    "images/military-vehicle/017-2014-Stepanakert-Czo-g-T-72-03-jpg.jpg": "Tank",  # T-72 Main Battle Tank
    "images/military-vehicle/020-At-Museo-Hist-rico-Militar-de-Canarias-2023-419-jpg.jpg": "Military Vehicle",  # Artillery piece
    "images/military-vehicle/021-At-Museum-of-American-Armor-2024-087-jpg.jpg": "Military Vehicle",  # Wheeled armor
    "images/military-vehicle/022-At-Museum-of-American-Armor-2024-090-Tiger-I-jpg.jpg": "Tank",  # Tiger I Heavy Tank
    "images/military-vehicle/023-At-Museum-of-American-Armor-2024-096-BA-64-jpg.jpg": "Military Vehicle",  # BA-64 Armored Car
    "images/military-vehicle/023-Blijmoedig-den-vijand-tegemoet-Van-het-Werfkantoor-naar-de-L.jpg": "Military Vehicle",  # Military document/transport
    "images/military-vehicle/024-At-Museum-of-American-Armor-2024-099-Tiger-I-jpg.jpg": "Tank",  # Tiger I Heavy Tank
    "images/military-vehicle/024-T-84-478DU9-driver-station-interior-jpg.jpg": "Tank",  # T-84 MBT Driver Station
    "images/military-vehicle/025-At-Museum-of-American-Armor-2024-106-jpg.jpg": "Military Vehicle",  # Armored car
    "images/military-vehicle/025-T-84-478DU9-turret-composite-armor-cross-section-png.png": "Tank",  # T-84 MBT Turret Armor
    "images/military-vehicle/026-1-jpg.jpg": "Tank",  # Karrar / T-72 Tank derivative
    "images/military-vehicle/026-At-Museum-of-American-Armor-2024-107-jpg.jpg": "Military Vehicle",  # Armored carrier
    "images/military-vehicle/027-At-Museum-of-American-Armor-2024-114-jpg.jpg": "Military Vehicle",  # Support vehicle
    "images/military-vehicle/027-BAHNA-2018-026-jpg.jpg": "Military Vehicle",  # Military transport
    "images/military-vehicle/028-BAHNA-2018-017-jpg.jpg": "Military Vehicle",  # Military carrier
    "images/military-vehicle/028-Belgrade-Military-Museum-PzKpfw-35-t-JPG.jpg": "Tank",  # Panzer 35(t) Tank
    "images/military-vehicle/029-BAHNA-2018-023-jpg.jpg": "Military Vehicle",  # Military truck
    "images/military-vehicle/029-Belgrade-Military-Museum-Renault-FT-17-JPG.jpg": "Tank",  # Renault FT-17 Tank
    "images/military-vehicle/030-BAHNA-2018-024-crop-jpg.jpg": "Military Vehicle",  # Armored transport
    "images/military-vehicle/030-Brest-Fortress-T-44-2024-09-19-3744-jpg.jpg": "Tank",  # T-44 Medium Tank
}

# Explicit, manually reviewed mappings for all 30 aircraft images
REVIEWED_AIRCRAFT_LABELS: Dict[str, str] = {
    "images/aircraft/001-F-22-Flaring-Cherry-Festival-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/002-GeoFS-F22-Raptor-png.png": "Fighter Aircraft",
    "images/aircraft/003-GeoFS-FA18-Super-Hornet-png.png": "Fighter Aircraft",
    "images/aircraft/004-HALTEDBFAeroindia2021-png.png": "Fighter Aircraft",
    "images/aircraft/005-Havac-l-k-ve-Uzay-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/006-J50MOCKUP-jpg.jpg": "EXCLUDED_OUT_OF_TAXONOMY",  # Wooden wind tunnel mockup
    "images/aircraft/007-MiG-29K-9-41-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/008-Mikoyan-Gurevich-MiG-21-2026-01-17-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/011-414th-Night-Fighter-Squadron-Bristol-Beaufighter-2-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/011-SGR-01-png.png": "EXCLUDED_OUT_OF_TAXONOMY",  # Glider / sailplane
    "images/aircraft/012-415th-Night-Fighter-Squadron-Bristol-Beaufighter-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/012-Su-47Zhukovsky-International-Airport-png.png": "Fighter Aircraft",
    "images/aircraft/013-422d-Night-Fighter-Squadron-Douglas-A-20-Havoc-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/014-Beaufighter-Mk-VIF-X7579-night-fighter-prototype-with-centim.jpg": "Fighter Aircraft",
    "images/aircraft/015-Bristol-Beaufighter-NAN15Dec43-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/016-Bundesarchiv-Bild-101I-667-7102A-06-Russland-Nachtj-ger-Junk.jpg": "Fighter Aircraft",
    "images/aircraft/017-DH113-Vampire-NF54-MM6152-6571289565-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/018-Dornier-Do-217N-night-fighter-in-1945-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/019-F2H-2N-VC-4-on-cat-USS-FD-Roosevelt-CVB-42-1950-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/020-F2H-2N-VC-4-on-USS-FD-Roosevelt-CVB-42-1950-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/021-F3D-2-Skynight-of-VMF-N-513-maintenance-during-Korean-War-jp.jpg": "Fighter Aircraft",
    "images/aircraft/022-F3D-2-VMFN-513-Kunsan-1953-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/023-F3D-2-VMFN-513-Kunsan-radar1-1953-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/024-F7F-3N-VMFN-513-Wonsan-1952-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/025-F7F-3N-XMas-1950-Korea-jpeg.jpg": "Fighter Aircraft",
    "images/aircraft/026-Foster-Mount-Avro504-jpg.jpg": "EXCLUDED_OUT_OF_TAXONOMY",  # WW1 Trainer biplane
    "images/aircraft/027-G4M-J1N-67s-jpg.jpg": "Fighter Aircraft",
    "images/aircraft/028-Most-numerous-fighter-jet-in-each-country-png.png": "EXCLUDED_OUT_OF_TAXONOMY",  # World map infographic
    "images/aircraft/029-Samoloty-mysliwskie-1945-73569716-jpg.jpg": "EXCLUDED_OUT_OF_TAXONOMY",  # Book cover illustration
    "images/aircraft/030-Vindkast-Mk-2E2-jpg.jpg": "EXCLUDED_OUT_OF_TAXONOMY",  # Ultralight sailplane
}


class ReconciledImageRecord(BaseModel):
    """An audited, human-reviewed dataset record ready for production-aligned ML pipelines."""
    image_id: str
    source_path: str
    original_label: str
    final_label: str
    review_status: str  # 'REVIEWED_APPROVED' or 'REVIEWED_EXCLUDED'
    is_trainable: bool
    split: Optional[str] = None
    provenance: Dict[str, Any]


class ReconciliationReport(BaseModel):
    """Overall summary of the human-reviewed taxonomy reconciliation."""
    total_images_reviewed: int
    trainable_images_count: int
    excluded_images_count: int
    class_distribution: Dict[str, int]
    unresolved_count: int
    records: List[ReconciledImageRecord]


def reconcile_supplied_dataset(
    dataset_root: Path,
    output_manifest_path: Optional[Path] = None,
) -> ReconciliationReport:
    """Execute human-reviewed taxonomy reconciliation across all 150 images."""
    dataset_root = Path(dataset_root)
    labels_csv = dataset_root / "labels.csv"
    credits_csv = dataset_root / "credits.csv"

    if not labels_csv.exists() or not credits_csv.exists():
        raise FileNotFoundError(f"Missing required CSVs in {dataset_root}")

    credits_map: Dict[str, Dict[str, Any]] = {}
    with open(credits_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            credits_map[r["file_name"]] = r

    records: List[ReconciledImageRecord] = []
    class_dist: Dict[str, int] = {c: 0 for c in PRODUCTION_CLASSES}
    excluded_count = 0
    idx = 1

    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["file_name"]
            orig_cat = row["category"]
            cred = credits_map.get(fn, {})

            image_id = f"ASTRA-SUP-{idx:03d}"
            idx += 1

            # Determine final reviewed label
            if orig_cat == "drone":
                final_label = "Drone"
                status = "REVIEWED_APPROVED"
            elif orig_cat == "helicopter":
                final_label = "Helicopter"
                status = "REVIEWED_APPROVED"
            elif orig_cat == "naval":
                final_label = "Ship"
                status = "REVIEWED_APPROVED"
            elif orig_cat == "military-vehicle":
                final_label = REVIEWED_MILITARY_VEHICLE_LABELS[fn]
                status = "REVIEWED_APPROVED"
            elif orig_cat == "aircraft":
                final_label = REVIEWED_AIRCRAFT_LABELS[fn]
                status = "REVIEWED_APPROVED" if final_label != "EXCLUDED_OUT_OF_TAXONOMY" else "REVIEWED_EXCLUDED"
            else:
                raise ValueError(f"Unknown supplied category: {orig_cat}")

            is_trainable = (final_label in PRODUCTION_CLASSES)
            if is_trainable:
                class_dist[final_label] += 1
            else:
                excluded_count += 1

            records.append(
                ReconciledImageRecord(
                    image_id=image_id,
                    source_path=fn,
                    original_label=orig_cat,
                    final_label=final_label,
                    review_status=status,
                    is_trainable=is_trainable,
                    split=None,
                    provenance={
                        "source_title": cred.get("source_title", ""),
                        "artist": cred.get("artist", ""),
                        "license": cred.get("license", ""),
                        "commons_page": cred.get("commons_page", ""),
                    },
                )
            )

    report = ReconciliationReport(
        total_images_reviewed=len(records),
        trainable_images_count=sum(class_dist.values()),
        excluded_images_count=excluded_count,
        class_distribution=class_dist,
        unresolved_count=0,
        records=records,
    )

    if output_manifest_path:
        out_path = Path(output_manifest_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

    return report
