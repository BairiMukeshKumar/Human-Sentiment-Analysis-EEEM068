# This script is a placeholder for the fusion model stage.
# The fusion model should be run only after both earlier visual models are completed:
# 1. Face-based model, trained using extracted face crops.
# 2. Full-image model, trained using complete MSCTD images.

# The actual fusion logic is prepared in:
# - src/fusion_utils.py: contains helper functions for combining model outputs/features.
# - src/models.py: contains the FusionMLP model class.

# The purpose of the fusion stage is to combine:
# - face-model prediction probabilities,
# - full-image model prediction probabilities,
# - number of detected faces,
# - optional face confidence information,
# into one final sentiment prediction.

print(
    "Fusion model placeholder: run after face model and full-image model are trained. "
    "src/fusion_utils.py and src/models.FusionMLP contain implementation hooks."
)