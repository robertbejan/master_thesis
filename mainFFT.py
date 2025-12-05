import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.models import SqueezeNet1_1_Weights
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import pandas as pd


class FFTTransform(torch.nn.Module):
    def __init__(self, use_magnitude=True, use_phase=False):
        super().__init__()
        self.use_magnitude = use_magnitude
        self.use_phase = use_phase

    def __call__(self, tensor):
        # Ensure tensor is in the right format (C, H, W)
        if tensor.dim() == 4:  # Batch dimension
            batch_fft = []
            for k in range(tensor.size(0)):
                batch_fft.append(self._apply_fft(tensor[k]))
            return torch.stack(batch_fft)
        else:
            return self._apply_fft(tensor)

    def _apply_fft(self, tensor):
        # Apply FFT to each channel
        fft_channels = []
        for c in range(tensor.size(0)):
            # Apply 2D FFT
            fft_result = torch.fft.fft2(tensor[c].float())

            channels_to_add = []
            if self.use_magnitude:
                magnitude = torch.abs(fft_result)
                # Apply log transform to compress dynamic range
                magnitude = torch.log(magnitude + 1e-8)
                channels_to_add.append(magnitude)

            if self.use_phase:
                phase = torch.angle(fft_result)
                channels_to_add.append(phase)

            fft_channels.extend(channels_to_add)

        return torch.stack(fft_channels)


def rgb_loader(path):
    """Load RGB image from various formats and convert to tensor"""
    if path.endswith('.npy'):
        # Load numpy array
        data = np.load(path, allow_pickle=True)

        # Ensure it's float32
        data = data.astype(np.float32)

        # If it's grayscale, convert to RGB by replicating channels
        if data.ndim == 2:
            data = np.stack([data, data, data], axis=-1)
        elif data.ndim == 3 and data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)

        # Ensure RGB format (H, W, 3)
        if data.ndim == 3 and data.shape[-1] == 3:
            # Convert to PIL Image
            # Normalize to 0-255 range if needed
            if data.max() <= 1.0:
                data = (data * 255).astype(np.uint8)
            else:
                data = np.clip(data, 0, 255).astype(np.uint8)

            return Image.fromarray(data, 'RGB')
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
    else:
        # Use default PIL loader for regular image files
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


# PARAMETERS

print('Program starting...')
input_size = (227, 227)  # Size of input images
batch_size = 30  # Mini-batch size
num_epochs = 20  # Number of epochs
learning_rate = 1e-4  # Learning rate
crop_left = 20
crop_right = 20
save_model_paths = [
    "../../mainProject/pythonProject1/rezCNNsqueezenetFilteredOrganized.pth",
    "../../mainProject/pythonProject1/rezCNNsqueezenetFilteredLarge.pth",
    "../../mainProject/pythonProject1/rezCNNsqueezenetFilteredSmall.pth"
]

train_paths = [
    "D:/Facultate/Disertatie/mainProject/pythonProject1/organized_ultrasound_dataset/labeled_train",
    "D:/Facultate/Disertatie/mainProject/pythonProject1/large_labeled_ultrasound_dataset/labeled_train",
    "D:/Facultate/Disertatie/mainProject/pythonProject1/small_labeled_ultrasound_dataset/labeled_train"
]

test_paths = [
    "D:/Facultate/Disertatie/mainProject/pythonProject1/organized_ultrasound_dataset/test",
    "D:/Facultate/Disertatie/mainProject/pythonProject1/large_labeled_ultrasound_dataset/test",
    "D:/Facultate/Disertatie/mainProject/pythonProject1/small_labeled_ultrasound_dataset/test"
]

exp_labels = ["50% labeled", "80% labeled", "20% labeled"]

# DATASET AND TRANSFORMS
# Data augmentation and normalization for train

for i in range(3):
    print(f"\n=== Experiment {i + 1} ({exp_labels[i]}) ===")
    train_data_path = train_paths[i]
    test_data_path = test_paths[i]
    save_model_path = save_model_paths[i]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            FFTTransform(use_magnitude=True, use_phase=False),  # Apply FFT transformation
            # Note: Normalization values might need adjustment for FFT data
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            FFTTransform(use_magnitude=True, use_phase=False),  # Apply FFT transformation
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    # Load the dataset with RGB loader
    image_dataset = datasets.DatasetFolder(train_data_path, loader=rgb_loader,
                                           extensions=('.npy', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
                                           transform=data_transforms['train'])

    # Split into train and validation datasets
    train_indices, val_indices = train_test_split(
        range(len(image_dataset)),
        test_size=0.3,
        stratify=image_dataset.targets,
        random_state=42,
    )
    train_dataset = torch.utils.data.Subset(image_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(image_dataset, val_indices)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get the number of classes
    num_classes = len(image_dataset.classes)

    # TEST DATASET
    test_dataset = datasets.DatasetFolder(test_data_path, loader=rgb_loader,
                                          extensions=('.npy', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
                                          transform=data_transforms['val'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Database loaded...')

    # LOAD PRE-TRAINED MODEL
    model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    print('Model loaded...')

    # Modify the classifier for the new dataset
    # Since we're using FFT magnitude of RGB (3 channels), we keep input as 3 channels
    # If using both magnitude and phase, change to 6 channels
    model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_classes

    # Transfer the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f'Training device set to {device}...')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # TRAINING
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")

    # VALIDATION
    print("Evaluating on validation data...")
    model.eval()
    val_labels, val_preds = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    accuracy_val = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {accuracy_val:.4f}")

    # TESTING
    print("Evaluating on test data...")
    test_labels, test_preds = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())

    accuracy_test = accuracy_score(test_labels, test_preds)
    print(f"Test Accuracy: {accuracy_test:.4f}")

    # Compute confusion matrices
    cm_val = confusion_matrix(val_labels, val_preds)
    cm_test = confusion_matrix(test_labels, test_preds)

    # Prepare Excel output
    excel_path = f"experiment_results_fft.xlsx"

    # Load or create workbook
    try:
        workbook = pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace')
    except FileNotFoundError:
        workbook = pd.ExcelWriter(excel_path, engine='openpyxl')

    sheet_name = f"Exp_{i + 1}_{exp_labels[i]}"

    # Build DataFrames
    df_summary = pd.DataFrame({
        "Metric": ["Validation Accuracy", "Test Accuracy"],
        "Value": [accuracy_val, accuracy_test]
    })

    df_cm_val = pd.DataFrame(cm_val)
    df_cm_val.index.name = "True"
    df_cm_val.columns.name = "Pred"

    df_cm_test = pd.DataFrame(cm_test)
    df_cm_test.index.name = "True"
    df_cm_test.columns.name = "Pred"

    # Write sheets
    df_summary.to_excel(workbook, sheet_name=sheet_name, startrow=0, index=False)
    df_cm_val.to_excel(workbook, sheet_name=sheet_name, startrow=5)
    df_cm_test.to_excel(workbook, sheet_name=sheet_name, startrow=5 + len(df_cm_val) + 4)

    # Save file
    workbook.close()

    print(f"Saved results to {excel_path}")
