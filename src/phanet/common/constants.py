"""Nine PhageScope Table S3 functional categories in model output order."""

CATEGORIES = (
    "Lysis", "Integration", "Replication", "tRNA-related", "Regulation",
    "Packaging", "Assembly", "Infection", "Immune",
)
CATEGORY_INDEX = {name: i for i, name in enumerate(CATEGORIES)}

# Default classification thresholds.
CATEGORY_THRESHOLDS = (0.5,) * len(CATEGORIES)
