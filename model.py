from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def unet(input_shape=(320, 320, 3), weights='imagenet',num_cls = 32):
    #加载ImageNet预训练权重
    vgg16_model = VGG16(input_shape=input_shape,
                        weights=weights,
                        include_top=False)

    # 编码过程VGG16经过4次降采样得到 20*20的特征图
    block4_pool = vgg16_model.get_layer('block4_pool').output
    block5_conv1 = Conv2D(1024, 3, activation='relu', padding='same',kernel_initializer='he_normal')(block4_pool)
    block5_conv2 = Conv2D(1024, 3, activation='relu', padding='same',kernel_initializer='he_normal')(block5_conv1)
    block5_drop = Dropout(0.5)(block5_conv2)

    # 第1次上采样解码，与VGG第3次降采样编码的结果融合得到 40*40的特征图
    block6_up = Conv2D(512, 2, activation='relu', padding='same',
                       kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(block5_drop))
    block6_merge = Concatenate(axis=3)([vgg16_model.get_layer('block4_conv3').output, block6_up])
    block6_conv1 = Conv2D(512, 3, activation='relu', padding='same',kernel_initializer='he_normal')(block6_merge)
    block6_conv2 = Conv2D(512, 3, activation='relu', padding='same',kernel_initializer='he_normal')(block6_conv1)
    block6_conv3 = Conv2D(512, 3, activation='relu', padding='same',kernel_initializer='he_normal')(block6_conv2)

    # 第2次上采样解码，与VGG第2次降采样编码的结果融合得到 80*80的特征图
    block7_up = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(block6_conv3))
    block7_merge = Concatenate(axis=3)([vgg16_model.get_layer('block3_conv3').output, block7_up])
    block7_conv1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_merge)
    block7_conv2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_conv1)
    block7_conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_conv2)

    # 第3次上采样解码，与VGG第1次降采样编码的结果融合得到 160*160的特征图
    block8_up = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(block7_conv3))
    block8_merge = Concatenate(axis=3)([vgg16_model.get_layer('block2_conv2').output, block8_up])
    block8_conv1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block8_merge)
    block8_conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block8_conv1)

    # 第4次上采样解码，与VGG降采样之前编码的结果融合得到 320*320的特征图
    block9_up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(block8_conv2))
    block9_merge = Concatenate(axis=3)([vgg16_model.get_layer('block1_conv2').output, block9_up])
    block9_conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_merge)
    block9_conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_conv1)
    block10_conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_conv2)

    # 将最后的特征图映射到像素分类空间中去，卷积输出通道数为像素类别数
    block10_conv2 = Conv2D(num_cls, 1, activation='sigmoid')(block10_conv1)

    model = Model(inputs=vgg16_model.input, outputs=block10_conv2)
    return model