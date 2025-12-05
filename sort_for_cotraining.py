import os
import shutil
import random
from sklearn.model_selection import train_test_split


def create_dataset_structure(base_dir, classes):
    """Create directory structure for the organized dataset"""
    dirs = [
        'labeled_train',
        'unlabeled_train',
        'validation',
        'test'
    ]

    for dir_name in dirs:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)
        # Create class subdirectories for labeled data
        if dir_name != 'unlabeled_train':
            for class_name in classes:
                os.makedirs(os.path.join(base_dir, dir_name, class_name), exist_ok=True)
        # Unlabeled data goes in a single directory
        else:
            os.makedirs(os.path.join(base_dir, dir_name, 'unlabeled'), exist_ok=True)


def organize_ultrasound_dataset(source_dirs, dest_dir, test_size=0.2, val_size=0.15, unlabeled_ratio=0.5):
    """
    Organize ultrasound images into labeled train, unlabeled train, validation, and test sets

    Args:
        source_dirs: List of directories containing original images (trainAUX, testAUX)
        dest_dir: Root directory where organized dataset will be saved
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        unlabeled_ratio: Proportion of training data to keep unlabeled
    """
    # First, collect all class names from the source directories
    class_names = set()
    for source_dir in source_dirs:
        for item in os.listdir(source_dir):
            if os.path.isdir(os.path.join(source_dir, item)):
                class_names.add(item)
    class_names = sorted(class_names)

    print(f"Found {len(class_names)} classes: {', '.join(class_names)}")

    # Create directory structure
    create_dataset_structure(dest_dir, class_names)

    # Collect all images with their class labels
    image_paths = []

    for source_dir in source_dirs:
        for class_name in class_names:
            class_dir = os.path.join(source_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append((os.path.join(class_dir, img_name), class_name))

    print(f"Found {len(image_paths)} total images across all classes")

    # Split into test and train sets
    train_paths, test_paths = train_test_split(
        image_paths,
        test_size=test_size,
        stratify=[x[1] for x in image_paths],
        random_state=42
    )

    # Split train into labeled and unlabeled
    labeled_paths, unlabeled_paths = train_test_split(
        train_paths,
        test_size=unlabeled_ratio,
        stratify=[x[1] for x in train_paths],
        random_state=42
    )

    # Split labeled into train and validation
    train_labeled_paths, val_paths = train_test_split(
        labeled_paths,
        test_size=val_size,
        stratify=[x[1] for x in labeled_paths],
        random_state=42
    )

    # Function to copy files to destination
    def copy_files(file_list, dest_subdir, keep_labels=True):
        for src_path, class_name in file_list:
            if keep_labels:
                dest_path = os.path.join(dest_dir, dest_subdir, class_name, os.path.basename(src_path))
            else:
                dest_path = os.path.join(dest_dir, dest_subdir, 'unlabeled', os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)

    # Copy files to their respective directories
    print("\nOrganizing dataset...")
    print("Copying labeled training data...")
    copy_files(train_labeled_paths, 'labeled_train')

    print("Copying unlabeled training data...")
    copy_files(unlabeled_paths, 'unlabeled_train', keep_labels=False)

    print("Copying validation data...")
    copy_files(val_paths, 'validation')

    print("Copying test data...")
    copy_files(test_paths, 'test')

    print("\nDataset organization complete!")
    print("=" * 50)
    print(f"{'Split':<20} {'Images':>10} {'% of Total':>15}")
    print("-" * 50)
    print(f"{'Labeled Train':<20} {len(train_labeled_paths):>10} {len(train_labeled_paths) / len(image_paths):>15.1%}")
    print(f"{'Unlabeled Train':<20} {len(unlabeled_paths):>10} {len(unlabeled_paths) / len(image_paths):>15.1%}")
    print(f"{'Validation':<20} {len(val_paths):>10} {len(val_paths) / len(image_paths):>15.1%}")
    print(f"{'Test':<20} {len(test_paths):>10} {len(test_paths) / len(image_paths):>15.1%}")
    print("=" * 50)
    print(f"\nOrganized dataset saved to: {dest_dir}")


# Configuration
source_dirs = [
    r"D:\Facultate\Disertatie\mainProject\pythonProject1\consolidated_dataset_simple\train",
    r"D:\Facultate\Disertatie\mainProject\pythonProject1\consolidated_dataset_simple\test"
]

destination_dir = r"D:\Facultate\Disertatie\mainProject\pythonProject1\large_labeled_ultrasound_dataset"

# Parameters (adjust these as needed)
test_size = 0.2  # 20% of data for test set
val_size = 0.15  # 15% of labeled training data for validation
unlabeled_ratio = 0.2  # 20% of training data to keep unlabeled

# Run the organization
if __name__ == "__main__":
    print("Starting ultrasound dataset organization...")
    organize_ultrasound_dataset(source_dirs, destination_dir, test_size, val_size, unlabeled_ratio)