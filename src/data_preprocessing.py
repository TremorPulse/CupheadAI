import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from object_detection import ObjectDetector


def load_and_preprocess_data(data_dir):
    X = []
    Y = []

    for file in os.listdir(data_dir):
        if file.endswith('.npy'):
            data = np.load(os.path.join(data_dir, file), mmap_mode='r')
            X.append(data['frames'])
            Y.append(data['actions'])

    # process data in chunks
    chunk_size = 1000
    for i in range(0, len(X), chunk_size):
        X_chunk = np.concatenate(X[i:i + chunk_size], axis=0)
        Y_chunk = np.concatenate(Y[i:i + chunk_size], axis=0)
        X_chunk = X_chunk / 255.0
        yield X_chunk, Y_chunk


def get_transform():
    return T.Compose([T.ToTensor()])


def get_dataloader(data_dir, batch_size=4):
    dataset = CocoDetection(root=os.path.join(data_dir, 'images'),
                            annFile=os.path.join(data_dir, '_annotations.coco.json'),
                            transform=get_transform())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


base_dir = r"C:\Users\gokua\Downloads\Cuphead Dataset.v2-cuphead-dataset.coco"
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

train_loader = get_dataloader(train_dir)
valid_loader = get_dataloader(valid_dir)

object_detector = ObjectDetector()
