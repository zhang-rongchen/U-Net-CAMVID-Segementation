import glob
import os
import numpy as np
import cv2
import tensorflow.keras.utils


class Generator(tensorflow.keras.utils.Sequence):
    def __init__(self, batch_size, shuffle,image_shape=(256,256), folder='train'):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_shape = image_shape
        self.x, self.y = load(folder,image_shape)
        self.indices = np.arange(len(self.x))

        assert len(self.x) == len(self.y)
        # assert len(self.x) % self.batch_size == 0

    def __getitem__(self, index):
        # if self.shuffle:
        #     self.indices = np.random.permutation(self.indices)
        i = index * self.batch_size
        indices = self.indices[i:i + self.batch_size]
        x = self.x[indices]
        y = self.y[indices]
        return (x, y)

    def __len__(self):
        return len(self.x) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indices = np.random.permutation(self.indices)

def load(folder,image_shape=(256,256)):

    # Load dataset.
    originals = []
    annotations = []
    for filename in map(lambda path: os.path.basename(path), glob.glob(f'./dataset/{folder}/*.png')):
        path1 = f'./dataset/{folder}/' + filename
        path2 = f'./dataset/{folder}annot/' + filename.split(".")[0]+"_P.png"

        image = cv2.imread(path1)
        image = cv2.resize(image,image_shape)
        image = np.float32(image) / 255.
        originals.append(image)

        image_label = cv2.imread(path2)[:, :, 0]
        image_label = cv2.resize(image_label, image_shape)
        labels = []
        for i in range(32):
            label = image_label == i
            label = np.asarray(label, np.int)

            labels.append(label)

        annotation = np.array(labels, dtype=np.int8)
        annotation = np.transpose(annotation, (1, 2, 0))
        annotations.append(annotation)
    annotations = np.array(annotations, dtype=np.float32)

    originals = np.array(originals, dtype=np.float32)


    return (originals, annotations)