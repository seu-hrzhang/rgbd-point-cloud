from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import numpy as np
import cv2
import os

config_file = "../configs/e2e_mask_rcnn_R_101_FPN_1x.yaml"

# Gray-scale colors used to identify objects
color_list = [255, 170, 85]
label_list = ["adgainai", "binghongcha250ml", "chunzhensuannai"]

# Update the config options with the config file
cfg.merge_from_file(config_file)
# Manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)


def balance_color(src):
    """
    :param src: source image with multiple masks
    :return: masks image with balanced color
    """
    list = color_list.copy()
    list.append(0)
    dst = np.zeros(src.shape, np.uint8)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i][j] == 0:
                continue
            for color in list:
                if abs(src[i][j] - color) < 5:
                    dst[i][j] = color
                    break
    return dst


def split_mask(src):
    """
    :param src: masks image
    :return: multiple binary mask images
    """
    scales = []
    for color in color_list:
        if color in src:
            scales.append(color)

    masks = []
    for scale in scales:
        mask = cv2.inRange(src, scale, scale)
        # cv2.imshow("Mask " + str(scale), mask)
        # cv2.waitKey()
        masks.append(mask)

    return scales, masks


def find_centroid(src):
    """
    :param src: input binary mask image
    :return: coordinate of object centroid
    """
    ctr = [0, 0]
    count = 0
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i][j] != 0:
                ctr = [a + b for a, b in zip(ctr, [j, i])]
                count += 1
    ctr = [round(a / b) for a, b in zip(ctr, [count, count])]
    return ctr


def get_2d_bbox(mask):
    bbox_img = mask.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    area = [cv2.contourArea(contour) for contour in contours]
    max_val = np.max(area)
    max_id = area.index(max_val)
    max_contour = contours[max_id]
    x, y, w, h = cv2.boundingRect(max_contour)

    # debug
    # cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (255, 255, 255), 5)
    # cv2.imshow("Bounding Box", bbox_img);
    # cv2.waitKey()

    return x, y, w, h


src_dir = '../datasets/rgb/'
src_list = os.listdir(src_dir)
mask_dir = "../../rgbd-point-cloud/mask/"

with open("../../rgbd-point-cloud/config/masks.txt", "r+") as file:
    file.truncate()

for src_name in src_list:
    src_id = src_name.strip('.jpg')
    src_path = src_dir + src_name
    src = cv2.imread(src_path)
    print("Running prediction for image " + src_id)
    mask, top_predictions = coco_demo.run_on_opencv_image(src)

    labels = top_predictions.get_field("labels").tolist()
    scores = top_predictions.get_field("scores").tolist()
    # save_path = mask_dir + src_name
    # cv2.imwrite(save_path, mask)

    # cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    # cv2.imshow("Mask", mask)
    # cv2.waitKey(1)

    gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    scales, sub_masks = split_mask(gray)
    print(str(len(sub_masks)) + " object(s) found.")

    with open("../../rgbd-point-cloud/config/masks.txt", "a") as file:
        for i, sub_mask in enumerate(sub_masks):
            print("Writing info of object " + str(i) + " in image " + src_id)
            scale_id = color_list.index(scales[i])
            label = label_list[scale_id]
            score = scores[labels.index(scale_id + 1)]
            # ctr = find_centroid(mask)
            x, y, w, h = get_2d_bbox(sub_mask)
            # cv2.circle(mask, ctr, 5, 0, cv2.FILLED)
            file.write(src_id + "\t" + str(i) + "\t")
            file.write(label + "\t")
            file.write(
                str(x) + "\t" + str(y) + "\t" + str(w) + "\t" + str(h) + "\t")
            file.write(str(score))
            file.write("\n")  # new line
            # file.write(str(ctr[0]) + "\t" + str(ctr[1]) + "\n")
            cv2.imwrite(mask_dir + src_id + "-" + str(i) + ".jpg", sub_mask)
        print("\n")

# print("Calling Point Cloud Proc Program...")
# os.system("point_cloud.exe")