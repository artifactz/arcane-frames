from typing import Iterable
from itertools import cycle, batched
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import db
from resnet_util import iter_frames_resnet
from quality_estimation.model import QualityModel
from quality_estimation.labels import load


def load_dataset(label_filenames: Iterable[str] | None = None) -> list[tuple[np.ndarray, float]]:
    # Load specified files or default file
    if label_filenames:
        data = {}
        for f in label_filenames:
            f_data = load(f)
            for video_filename, video_labels in f_data.items():
                data.setdefault(video_filename, {})
                data[video_filename] = data[video_filename] | video_labels
    else:
        data = load()

    # Remove null labels
    for video_filename, video_labels in data.items():
        for k in list(video_labels.keys()):
            if video_labels[k] is None:
                del video_labels[k]

    # Get and concat resnet vectors
    return [
        (np.concat((embed, prob)), y)
        for video_filename, ratings in data.items()
        for (embed, prob), y in zip(iter_frames_resnet(video_filename, list(ratings.keys())), ratings.values())
    ]


def split_train_test(dataset: np.ndarray, test_ratio=0.1):
    n = round(test_ratio * len(dataset))
    return dataset[:n], dataset[n:]


def train(dataset: list):
    model = QualityModel(in_features=3048, out_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Prepare Data
    np.random.shuffle(dataset)
    xs = np.array([x[0] for x in dataset], dtype=np.float32)
    ys = np.array([x[1] for x in dataset], dtype=np.float32)
    ys -= ys.mean()
    ys /= ys.std()

    test_xs, train_xs = split_train_test(xs)
    test_ys, train_ys = split_train_test(ys)
    test_xs = torch.from_numpy(np.array(test_xs))
    test_ys = torch.from_numpy(np.array(test_ys)[:, np.newaxis])

    # Training Loop
    batch_size = 64
    total_epochs = 100
    batches_per_epoch = len(train_xs) / batch_size
    epoch = 0
    n_batches = 0
    loss = None
    test_loss = None

    for batch_xs, batch_ys in zip(batched(cycle(train_xs), batch_size), batched(cycle(train_ys), batch_size)):
        if n_batches % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_output = model(test_xs)
                test_loss = criterion(test_output, test_ys)
            model.train()
            scheduler.step(test_loss)

            train_loss_str = f'{loss.item():.4f}' if loss else 'N/A'
            test_loss_str = f'{test_loss.item():.4f}'
            print(f'Epoch [{epoch:.2f}/{total_epochs}], Loss: {train_loss_str}, Test Loss: {test_loss_str}')

        # Forward pass
        output = model(torch.from_numpy(np.array(batch_xs)))
        loss = criterion(output, torch.from_numpy(np.array(batch_ys)[:, np.newaxis]))

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights

        n_batches += 1
        epoch = n_batches / batches_per_epoch
        if epoch >= total_epochs:
            break

    torch.save(model.state_dict(), "quality_estimation/model.pth")


if __name__ == "__main__":
    db.ensure_frame_resnet_table()
    dataset = load_dataset()
    train(dataset)
