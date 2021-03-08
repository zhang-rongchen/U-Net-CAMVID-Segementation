import numpy as np
import cv2
from model import unet
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class_colors = [(64,128,64), (192,0,128), (0,128,192),(0,128,64),(128,0,0),(64,0,128),(64,0,192),(192,128,64),
                (192,192,128),(64,64,128),(128,0,192),(192,0,64),(128,128,64),(192,0,192),(128,64,64),(64,192,128),
                (64,64,0),(128,64,128),(128,128,192),(0,0,192),(192,128,128),(128,128,128),(64,128,192),(0,0,64),
                (0,64,64),(192,64,128),(128,128,0),(192,128,192),(64,0,64),(192,192,0),(0,0,0),(64,192,0)]

class_names = ["Animal","Archway","Bicyclist","Bridge","Building","Car","CartLuggagePram","Child","Column_Pole",
               "Fence","LaneMkgsDriv","LaneMkgsNonDriv","Misc_Text","MotorcycleScooter","OtherMoving","ParkingBlock",
               "Pedestrian","Road","RoadShoulder","Sidewalk","SignSymbol","Sky","SUVPickupTruck","TrafficCone",
               "TrafficLight","Train","Tree","Truck_Bus","Tunnel","VegetationMisc","Void","Wall"]


EPS = 1e-12


def get_iou(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou
    return class_wise

def get_acc(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou
    return class_wise





def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img

def get_legends(class_names, colors=class_colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend
def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img


def concat_lenends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img

def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False):

    if n_classes is None:
        n_classes = np.max(seg_arr)

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)

        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img

def main():
    # 构建模型.
    model = unet(input_shape=(320, 320,3),num_cls=32)
    # 加载模型
    model.load_weights("weights/unet_camvid_weights.h5")
    sumiou = 0
    sumacc = 0
    root = "dataset/test/"
    testList = os.listdir("dataset/test/")

    # 数据
    for file in testList:
        if file.endswith("png"):
            labelfile = "C:\\Users\\41648\\Downloads\\camvid-master\\camvid-master\\LabeledApproved_full\\" + \
                        file.split(".")[0] + "_L.png"
            gtfile = "dataset/testannot/"+file.split(".")[0] + "_P.png"
            image = cv2.imread(os.path.join(root,file))
            image = cv2.resize(image, (320,320))
            image_ = np.float32(image) / 255.
            batch_test_img = np.expand_dims(image_, axis=0)
            pred = model.predict(batch_test_img)[0]

            gt = cv2.imread(gtfile,0)
            gt = cv2.resize(gt, (320, 320))
            pr = pred.argmax(axis=2)
            sumacc += np.sum((gt == pr))/(320*320)
            iouCls = np.zeros((32, 2))
            for cl in range(32):
                if np.sum((gt == cl))>0:
                    intersection = np.sum((gt == cl) * (pr == cl))
                    union = np.sum(np.maximum((gt == cl), (pr == cl)))
                    iou = float(intersection) / (union + EPS)
                    iouCls[cl][0] += 1
                    iouCls[cl][1] += iou
            # print(iouCls)
            sumiou += iouCls
            seg_img = visualize_segmentation(pr, image, n_classes=32,
                                     colors=class_colors, overlay_img=False,
                                     show_legends=False,
                                     class_names=class_names)

            seg_img = np.asarray(seg_img,np.uint8)
            seg_img = cv2.cvtColor(seg_img,cv2.COLOR_RGB2BGR)

            label = cv2.imread(labelfile)
            label = cv2.resize(label,(320,320))

            out = np.hstack([label,image,seg_img])
            # cv2.imwrite("result/"+file,out)
            cv2.imshow("seg_img",out)
            cv2.waitKey(50)

if __name__ == '__main__':
    main()
