import os
import random
import shutil
from tqdm import tqdm


def prepare_oct_data():
    """
    Prepares the OCT2017 dataset by splitting it into balanced training data, 
    validation data, and an upsampled test set. It creates the necessary folders
    and copies the images accordingly.
    """
    # Define the paths to the data folders
    data_folder = '/Users/youssefshaarawy/Documents/Datasets/OCT2017'
    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'test')
    val_folder = os.path.join(data_folder, 'val')
    train_balanced_folder = os.path.join(data_folder, 'train_balanced')
    new_test_folder = os.path.join(data_folder, 'test_upsampled')

    # Clean up any existing data in the validation, balanced train, and upsampled test folders
    for folder in [val_folder, train_balanced_folder, new_test_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # Parameters for splitting the data
    val_class_size = 250  # Number of validation images per class
    extra_test_images_size = 250  # Number of additional test images per class
    train_class_size = 11348 - val_class_size - extra_test_images_size  # Remaining images for training

    # Process each class folder within the training data
    for class_folder in tqdm(os.listdir(train_folder)):
        class_path = os.path.join(train_folder, class_folder)
        if not os.path.isdir(class_path):
            continue  # Skip if not a directory

        # Get all images from the class folder and shuffle them randomly
        class_images = os.listdir(class_path)
        random.shuffle(class_images)

        # Split the images into training, validation, and extra test sets
        extra_test_images = class_images[
            -extra_test_images_size:]  # Last n images for test
        val_images = class_images[-(
            val_class_size + extra_test_images_size
        ):-extra_test_images_size]  # Just before the test images for validation

        # If there are more images than needed for training, use the specified training size
        if len(class_images) > train_class_size:
            train_images = class_images[:train_class_size]
        else:
            train_images = class_images[:-(val_class_size +
                                           extra_test_images_size)]

        # Print the summary of images for each class
        print(
            f"{class_folder}: {len(train_images)} train, {len(val_images)} val, {len(extra_test_images) + len(os.listdir(os.path.join(test_folder, class_folder)))} test"
        )

        # Create class-specific subdirectories in the balanced train, validation, and new test folders
        os.makedirs(os.path.join(train_balanced_folder, class_folder),
                    exist_ok=True)
        os.makedirs(os.path.join(val_folder, class_folder), exist_ok=True)
        os.makedirs(os.path.join(new_test_folder, class_folder), exist_ok=True)

        # Copy the images to the respective directories
        for img in train_images:
            shutil.copy(os.path.join(train_folder, class_folder, img),
                        os.path.join(train_balanced_folder, class_folder, img))
        for img in val_images:
            shutil.copy(os.path.join(train_folder, class_folder, img),
                        os.path.join(val_folder, class_folder, img))
        for img in extra_test_images:
            shutil.copy(os.path.join(train_folder, class_folder, img),
                        os.path.join(new_test_folder, class_folder, img))

        # Add the existing test images to the new upsampled test set with modified filenames
        for img in os.listdir(os.path.join(test_folder, class_folder)):
            shutil.copy(
                os.path.join(test_folder, class_folder, img),
                os.path.join(new_test_folder, class_folder,
                             os.path.splitext(img)[0] + '_train.jpg'))


if __name__ == '__main__':
    prepare_oct_data()
