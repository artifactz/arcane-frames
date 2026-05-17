import json


DEFAULT_LABELS_FILENAME = "quality_estimation/quality_labels.json"


def save(labels: dict[str, dict[int, int]], filename=DEFAULT_LABELS_FILENAME):
    with open(filename, "w") as f:
        json.dump(labels, f, indent=2)

def load(filename=DEFAULT_LABELS_FILENAME) -> dict[str, dict[int, int]]:
    with open(filename, "r") as f:
        labels = json.load(f)
    return {
        # Convert frame index keys back to int (json has str keys)
        filename: {int(frame_index): rating for frame_index, rating in file_labels.items()}
        for filename, file_labels in labels.items()
    }
