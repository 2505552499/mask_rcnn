import os
import json
import random
import shutil
from collections import defaultdict

def split_dataset(
    image_dir,
    annotation_path,
    train_image_dir,
    val_image_dir,
    train_annotation_path,
    val_annotation_path,
    split_ratio=0.8,
    seed=42
):
    """
    Splits BMP images and corresponding COCO annotations into training and validation sets.

    Parameters:
    - image_dir (str): Path to the directory containing BMP images.
    - annotation_path (str): Path to the COCO format JSON annotation file.
    - train_image_dir (str): Destination directory for training images.
    - val_image_dir (str): Destination directory for validation images.
    - train_annotation_path (str): Path to save the training annotations JSON.
    - val_annotation_path (str): Path to save the validation annotations JSON.
    - split_ratio (float): Proportion of data to be used for training.
    - seed (int): Random seed for reproducibility.
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Ensure destination directories exist
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)

    # 1. Load all BMP images
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith('.bmp')]
    total_images = len(all_images)
    print(f"Total BMP images found: {total_images}")

    # 2. Load COCO annotations
    with open(annotation_path, 'r') as f:
        coco = json.load(f)

    # Create a mapping from file name to image info
    filename_to_image = {img['file_name']: img for img in coco['images']}
    image_id_to_image = {img['id']: img for img in coco['images']}

    # 3. Filter out images without annotations
    annotated_filenames = set(filename_to_image.keys())
    annotated_images = [img for img in all_images if img in annotated_filenames]
    total_annotated = len(annotated_images)
    print(f"Total annotated images: {total_annotated}")

    if total_annotated != 1984:
        print(f"Warning: Expected 1984 annotated images, but found {total_annotated}.")

    # 4. Shuffle and split the annotated images
    random.shuffle(annotated_images)
    split_index = int(split_ratio * total_annotated)
    train_filenames = annotated_images[:split_index]
    val_filenames = annotated_images[split_index:]
    print(f"Training images: {len(train_filenames)}")
    print(f"Validation images: {len(val_filenames)}")

    # 5. Create sets of image IDs
    train_image_ids = set(filename_to_image[filename]['id'] for filename in train_filenames)
    val_image_ids = set(filename_to_image[filename]['id'] for filename in val_filenames)

    # 6. Split annotations
    def filter_annotations(coco, image_ids):
        return {
            key: [ann for ann in coco[key] if ann['image_id'] in image_ids]
            for key in ['annotations']
        }

    train_annotations = filter_annotations(coco, train_image_ids)
    val_annotations = filter_annotations(coco, val_image_ids)

    # 7. Split images
    train_images = [filename_to_image[filename] for filename in train_filenames]
    val_images = [filename_to_image[filename] for filename in val_filenames]

    # 8. Prepare final COCO JSON structures
    def prepare_coco_json(coco, images, annotations):
        return {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": images,
            "annotations": annotations['annotations'],
            "categories": coco.get("categories", [])
        }

    train_coco = prepare_coco_json(coco, train_images, train_annotations)
    val_coco = prepare_coco_json(coco, val_images, val_annotations)

    # 9. Save the new annotation JSON files
    with open(train_annotation_path, 'w') as f:
        json.dump(train_coco, f, indent=4)
    print(f"Training annotations saved to {train_annotation_path}")

    with open(val_annotation_path, 'w') as f:
        json.dump(val_coco, f, indent=4)
    print(f"Validation annotations saved to {val_annotation_path}")

    # 10. Function to copy images
    def copy_images(filenames, src_dir, dest_dir):
        for filename in filenames:
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy(src_path, dest_path)

    # 11. Copy images to respective directories
    copy_images(train_filenames, image_dir, train_image_dir)
    copy_images(val_filenames, image_dir, val_image_dir)
    print("Images have been copied to training and validation directories.")

    # 12. Validate the sum of image_ids
    total_image_ids = len(train_image_ids) + len(val_image_ids)
    expected_total = len(coco['images'])
    if total_image_ids != expected_total:
        print(f"Warning: Sum of train and val image_ids ({total_image_ids}) does not equal total annotated images ({expected_total}).")
    else:
        print("Validation successful: Sum of image_ids matches the original annotations.")

if __name__ == "__main__":
    # Define paths
    IMAGE_DIR = 'datasets/images/train/'
    ANNOTATION_PATH = 'datasets/images/annotations/instances_train.json'
    TRAIN_IMAGE_DIR = 'datasets/images/train_set/'
    VAL_IMAGE_DIR = 'datasets/images/val_set/'
    TRAIN_ANNOTATION_PATH = 'datasets/images/annotations/instances_train.json'
    VAL_ANNOTATION_PATH = 'datasets/images/annotations/instances_val.json'

    split_dataset(
        image_dir=IMAGE_DIR,
        annotation_path=ANNOTATION_PATH,
        train_image_dir=TRAIN_IMAGE_DIR,
        val_image_dir=VAL_IMAGE_DIR,
        train_annotation_path=TRAIN_ANNOTATION_PATH,
        val_annotation_path=VAL_ANNOTATION_PATH,
        split_ratio=0.8,
        seed=42
    )