import logging
import os

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    base_dir = '/content/drive/MyDrive/Cuphead Dataset.v2-cuphead-dataset.coco'
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')

    # debugging to check if directories exist
    logging.info(f"Base directory exists: {os.path.exists(base_dir)}")
    logging.info(f"Train directory exists: {os.path.exists(train_dir)}")
    logging.info(f"Valid directory exists: {os.path.exists(valid_dir)}")
    train_ann_file = os.path.join(train_dir, '_annotations.coco.json')
    valid_ann_file = os.path.join(valid_dir, '_annotations.coco.json')
    logging.info(f"Train annotation file exists: {os.path.exists(train_ann_file)}")
    logging.info(f"Valid annotation file exists: {os.path.exists(valid_ann_file)}")
    logging.info(f"Contents of train directory: {os.listdir(train_dir)}")
    logging.info(f"Contents of valid directory: {os.listdir(valid_dir)}")
