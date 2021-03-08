from model import unet
import os
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION']='1'
from data import Generator
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf


def loss(y_true, y_pred):
    smooth = 1.
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
    return binary_crossentropy(y_true, y_pred)+0.1*(1.-dice_coef(y_true, y_pred))

def main():
    batch_size = 8
    Epochs = 50
    learning_rate = 0.0005
    weights_path = 'weights/'
    directory = 'logs/'
    img_size = 320
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    # Prepare model.
    model = unet(input_shape=(img_size, img_size,3), num_cls=32)
    model.summary()

    model.compile(optimizer=opt, loss=binary_crossentropy, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(
        weights_path + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-val_accuracy{val_accuracy:.3f}1.h5',
        monitor='val_accuracy', save_weights_only=True, save_best_only=True, mode='max', period=1)
    tensorboard = TensorBoard(log_dir=directory, batch_size=batch_size)
    callbacks = [checkpoint, tensorboard]

    # Training.
    model.fit_generator(
            generator=Generator(batch_size=batch_size,image_shape=(img_size, img_size),shuffle=True,folder = "train"),
            steps_per_epoch=int(473/batch_size),
            validation_data = Generator(batch_size=batch_size, image_shape=(img_size, img_size),shuffle=False, folder="val"),
            validation_steps=int(162/batch_size),
            epochs=Epochs,
            shuffle=True,verbose=1,callbacks=callbacks)

if __name__ == '__main__':
    main()