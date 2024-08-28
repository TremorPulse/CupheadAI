import logging
import os

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Set up paths
    base_dir = '/content/drive/MyDrive/Cuphead Dataset.v2-cuphead-dataset.coco'
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')

    # Debug: Check if directories exist
    logging.info(f"Base directory exists: {os.path.exists(base_dir)}")
    logging.info(f"Train directory exists: {os.path.exists(train_dir)}")
    logging.info(f"Valid directory exists: {os.path.exists(valid_dir)}")

    # Debug: Check if annotation files exist
    train_ann_file = os.path.join(train_dir, '_annotations.coco.json')
    valid_ann_file = os.path.join(valid_dir, '_annotations.coco.json')
    logging.info(f"Train annotation file exists: {os.path.exists(train_ann_file)}")
    logging.info(f"Valid annotation file exists: {os.path.exists(valid_ann_file)}")

    # Debug: List contents of train and valid directories
    logging.info(f"Contents of train directory: {os.listdir(train_dir)}")
    logging.info(f"Contents of valid directory: {os.listdir(valid_dir)}")

    # Initialize the ObjectDetector
    # detector = ObjectDetector()

    # Create dataloaders
    #train_loader = detector.get_dataloader(train_dir)
    # valid_loader = detector.get_dataloader(valid_dir)

    # Train the model
    # detector.train(train_loader, valid_loader)
