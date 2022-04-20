#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:34:46 2018

@author: avsthiago
"""

import numpy as np
import cv2
import os
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import math
from tqdm import tqdm
from collections import Counter
import datetime
import warnings
import imghdr
from pathlib import PurePath

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PATH = os.path.dirname(os.path.realpath("__file__"))

PATH_IMAGES = os.path.join(*list(PurePath("../original_images/").parts))
PATH_MODEL = "model"
PATH_DETECTIONS = os.path.join(*list(PurePath("../annotations/detections/").parts))
PATH_PREDICTIONS = os.path.join(*list(PurePath("../annotations/predictions/").parts))
PATH_OUT_IMAGE = os.path.join(*list(PurePath("../output/labeled_images/").parts))
PATH_OUT_CSV = os.path.join(*list(PurePath("../output/spreadsheet/").parts))
MIN_CONFIDENCE = 0.9995

LEFT_BAR_SIZE = 480
img_size = 224
batch_size = 100


def cross_plataform_directory():
    global PATH_IMAGES, PATH_DETECTIONS, PATH_PREDICTIONS, PATH_OUT_IMAGE, PATH_OUT_CSV
    if "\\" in PATH_IMAGES:
        PATH_IMAGES += "\\"
        PATH_DETECTIONS += "\\"
        PATH_PREDICTIONS += "\\"
        PATH_OUT_IMAGE += "\\"
        PATH_OUT_CSV += "\\"
    elif "/" in PATH_IMAGES:
        PATH_IMAGES += "/"
        PATH_DETECTIONS += "/"
        PATH_PREDICTIONS += "/"
        PATH_OUT_IMAGE += "/"
        PATH_OUT_CSV += "/"


def get_qtd_by_class(points, labels):
    points_filtered = points[points[:, 4] == 1, 3]
    sum_predictions = Counter(points_filtered)
    return [
        *[str(sum_predictions[i]) for i, j in enumerate(labels)],
        str(len(points_filtered)),
    ]


def get_header(labels):
    return "Img Name," + ",".join([i for i in labels]) + ",Total\n"


def draw_labels_bar(image, labels, colors):
    height = image.shape[0]
    left_panel = np.zeros((height, LEFT_BAR_SIZE, 3), dtype=np.uint8)
    labels = [l.title() for l in labels]

    for i, cl in enumerate(zip(colors, labels)):
        color, label = cl
        cv2.putText(
            left_panel,
            " ".join([str(i + 1), ".", label]),
            (15, 70 * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.4,
            color,
            2,
        )

    return np.hstack((left_panel, image))


def draw_circles_labels(image, labels, points, colors=None, draw_labels=True):
    if colors is None:

        colors = [
            (255, 0, 0),
            (0, 255, 255),
            (0, 0, 128),
            (255, 0, 255),
            (0, 255, 0),
            (255, 255, 100),
            (0, 0, 255),
        ]

    if draw_labels:
        image = draw_labels_bar(np.copy(image), labels, colors)

    points[:, 0] += LEFT_BAR_SIZE

    for p in points:
        cv2.circle(image, (p[0], p[1]), p[2], colors[p[3]], 4)

    points[:, 0] -= LEFT_BAR_SIZE
    return image


def extract_circles(
    image, pts, output_size=224, mean_radius_default=32, standardize_radius=True
):
    """
    extract cells from a image:
    Parameters
    ----------
    image : image with full size
    pts : ndarray with a set of points in the shape [W, H, R] R stands for
          radius
    output_size : all images will be returned with the size
                  (output_size, output_size)
    mean_radius_default : if standardize_radius is set, thes parameter will be
                          used as a base size to resize all circle detections
                          32 is the average radius of a cell
    Returns
    -------
    ROIs : (N x W x H x C) N as the total number of detections and K is the
           number of channels
    """
    if standardize_radius:
        # use the mean radius to calculate the clip size to each detection
        pts[:, 2] = output_size / mean_radius_default * pts[:, 2]
        # the border needs to be greater than the biggest clip
        size_border = pts[:, 2].max() + 1
        # deslocates the detection centers
        pts[:, [0, 1]] = pts[:, [0, 1]] + size_border

        # creates a border around the main image
        img_w_border = cv2.copyMakeBorder(
            image,
            size_border,
            size_border,
            size_border,
            size_border,
            cv2.BORDER_REFLECT,
        )

        # extracts all detections and resizes them
        ROIs = [
            cv2.resize(
                img_w_border[i[1] - i[2] : i[1] + i[2], i[0] - i[2] : i[0] + i[2]],
                (224, 224),
            )
            for i in pts
        ]

    return ROIs


def classify_image(im_name, npy_name, labels, net, img_size, file):
    try:
        if not os.path.isfile(im_name):
            raise
        if not os.path.isfile(npy_name):
            raise

        image = cv2.imread(im_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        points = np.load(npy_name)

        pt = np.copy(points)
        pt[:, 2] = pt[:, 2] // 2

        blob_imgs = extract_circles(image, np.copy(pt), output_size=img_size)
        blob_imgs = np.asarray([i for i in blob_imgs])
        blob_imgs = preprocess_input(blob_imgs)

        scores = None

        for chunk in [
            blob_imgs[x : x + batch_size] for x in range(0, len(blob_imgs), batch_size)
        ]:
            output = net.predict(chunk)

            if scores is None:
                scores = np.copy(output)
            else:
                scores = np.vstack((scores, output))

        lb_predictions = np.argmax(scores, axis=1)
        vals_predictions = np.amax(scores, axis=1)

        points_pred = np.hstack(
            (np.copy(points), np.expand_dims(lb_predictions, axis=0).T)
        )

        sum_predictions = Counter(lb_predictions)
        lb = [j + " " + str(sum_predictions[i]) for i, j in enumerate(labels)]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_predita = draw_circles_labels(image, lb, points_pred)

        inside_roi = np.ones_like(points_pred[:, 3])
        new_class = np.copy(points_pred[:, 3])

        st_use_retrain = (vals_predictions > MIN_CONFIDENCE) * 1

        csl = np.vstack(
            [i for i in [new_class, st_use_retrain, inside_roi, vals_predictions]]
        ).T

        points_pred = np.hstack((points_pred, csl))

        if file is not None:
            file.write(
                ",".join(
                    [im_name.split("/")[-1], *get_qtd_by_class(points_pred, labels)]
                )
                + "\n"
            )

        date_saved = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
        height, width, _ = image.shape
        roi = ((0, 0), (width, height))

        array_to_save = np.array([roi, date_saved, points_pred])

        if PurePath(im_name.replace(PATH_IMAGES, "")).parts[:-1]:
            dest_folder = os.path.join(
                PATH_PREDICTIONS,
                os.path.join(*PurePath(im_name.replace(PATH_IMAGES, "")).parts[:-1]),
            )
        else:
            dest_folder = PATH_PREDICTIONS

        array_name = PurePath(im_name).parts[-1].split(".")[:-1][0] + ".npy"
        array_name = os.path.join(dest_folder, array_name)

        create_folder(array_name)
        np.save(array_name, array_to_save)

        out_img_name = os.path.join(PATH_OUT_IMAGE, im_name.replace(PATH_IMAGES, ""))
        create_folder(out_img_name)
        cv2.imwrite(out_img_name, cv2.resize(img_predita, (1500, 1000)))
    except Exception as e:
        print("\nFiled to classify image " + im_name, e)


def segmentation(img, model):
    IMG_WIDTH_DEST = 482
    IMG_HEIGHT_DEST = 482
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    original_shape = img.shape[:2]

    if original_shape != (4000, 6000):
        img = cv2.resize(img, (6000, 4000))

    reflect = cv2.copyMakeBorder(img, 184, 184, 148, 148, cv2.BORDER_REFLECT)

    pos_x = np.arange(0, 5785, 482)
    pos_y = np.arange(0, 3857, 482)
    slices = [
        np.s_[y[0] : y[1], x[0] : x[1]]
        for x in zip(pos_x, pos_x + 512)
        for y in zip(pos_y, pos_y + 512)
    ]

    X = np.zeros((len(slices), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    for j, sl in enumerate(slices):
        X[j] = cv2.resize(
            reflect[sl], (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA
        )

    preds = model.predict(X)
    preds = (preds > 0.5).astype(np.uint8)

    RESULT_Y = np.zeros(
        (len(preds), IMG_HEIGHT_DEST, IMG_WIDTH_DEST, 1), dtype=np.uint8
    )

    for j, x in enumerate(preds):
        RESULT_Y[j] = np.expand_dims(
            cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)[15:497, 15:497],
            axis=-1,
        )

    reconstructed_mask = (
        np.squeeze(np.hstack([np.vstack(i) for i in np.split(RESULT_Y, 13)]))[
            169:4169, 133:6133
        ]
        * 255
    )

    if original_shape != (4000, 6000):
        reconstructed_mask = cv2.resize(
            reconstructed_mask, (original_shape[1], original_shape[0])
        )

    # remove internal areas
    _, contours, _ = cv2.findContours(reconstructed_mask, 1, 2)
    max_cnt = contours[np.argmax(np.array([cv2.contourArea(i) for i in contours]))]

    reconstructed_mask *= 0
    cv2.drawContours(reconstructed_mask, [max_cnt], 0, (255, 255, 255), -1)

    bounding_rect = cv2.boundingRect(max_cnt)  # x,y,w,h

    return reconstructed_mask, bounding_rect


def find_circles(im_name, img, mask, cnt):
    try:
        x, y, w, h = cnt

        roi = np.copy(img[y : y + h, x : x + w])
        roi = cv2.split(roi)[2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
        roi = clahe.apply(roi)
        roi = cv2.bilateralFilter(roi, 5, 50, 50)

        # find all cells with different radius
        all_points = np.array([])
        for j in range(5, 50, 5):
            points = cv2.HoughCircles(
                roi,
                cv2.HOUGH_GRADIENT,
                dp=2,
                minDist=12,
                param1=145,
                param2=55,
                minRadius=j + 1,
                maxRadius=j + 5,
            )

            if points is not None:
                points = points[0][:, :3].astype(np.int32)
                all_points = (
                    np.vstack((all_points, points)) if all_points.size else points
                )

        # select best radius
        if all_points.size == 0:
            best_radius = 33
        else:
            best_radius = np.bincount(all_points[:, -1]).argmax()

        minDist = best_radius * 2 - ((best_radius * 9 / 26) + 75 / 26)

        minRadius = best_radius - max(2, math.floor(best_radius * 0.1))
        maxRadius = best_radius + max(2, math.floor(best_radius * 0.1))

        # hough to find all cells
        points = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=3,
            minDist=minDist,
            param1=100,
            param2=25,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )

        if points is not None:
            points = points[0][:, :3]
            points[:, 2:] = points[:, 2:]
            points = points.astype(np.int32)
            points = points[points[:, 0] < w]
            points = points[points[:, 1] < h]

            points[:, 0] += x
            points[:, 1] += y

            points = points[mask[points[:, 1], points[:, 0]] > 0]

        np_name = [PurePath(im_name).parts[-1].split(".")[:-1][0] + ".npy"]
        array_name = os.path.join(
            *[PATH_DETECTIONS] + list(PurePath(im_name).parts[:-1]) + np_name
        )
        create_folder(array_name)
        np.save(array_name, points)
    except:
        print("Cell detection failed on image ", PurePath(im_name).parts[-1] + "\n")


def create_folder(path):
    path = os.path.join(*PurePath(path).parts[:-1])
    if not os.path.exists(path):
        os.makedirs(path)


def find_image_names():
    l_images = []
    for path, subdirs, files in os.walk(PATH_IMAGES):
        for name in files:
            full_path = os.path.join(path, name)
            if imghdr.what(full_path) is not None:
                l_images.append(full_path.replace(PATH_IMAGES, ""))
    return l_images


def create_detections():
    dic_model = load_dict_model(PATH_MODEL)
    images = find_image_names()
    m_border = load_model(dic_model["border"])

    with tqdm(total=len(images)) as j:
        for i in images:
            img = cv2.imread(os.path.join(PATH_IMAGES, i))
            mask, cnt = segmentation(img, m_border)
            find_circles(i, img, mask, cnt)
            j.update(1)


def load_dict_model(path):
    # gets all files inside the path
    files = os.listdir(path)
    model = {}

    model["model_h5"] = os.path.join(
        path, list(filter(lambda x: "classification" in x, files))[0]
    )

    model["border"] = os.path.join(
        path, list(filter(lambda x: "segmentation" in x, files))[0]
    )

    model["labels"] = ["Capped", "Eggs", "Honey", "Larves", "Nectar", "Other", "Pollen"]
    return model


def classify_images():
    images = sorted([os.path.join(PATH_IMAGES, i) for i in find_image_names()])

    find_image_detections = lambda i: ".".join(i.split(".")[:-1]) + ".npy"

    detections = [
        os.path.join(PATH_DETECTIONS, find_image_detections(i).replace(PATH_IMAGES, ""))
        for i in images
    ]

    dict_model = load_dict_model(PATH_MODEL)
    net = load_model(dict_model["model_h5"])

    with tqdm(total=len(images)) as t:
        for i, j in zip(images, detections):
            classify_image(i, j, dict_model["labels"], net, img_size, None)
            t.update(1)


def main():
    cross_plataform_directory()
    print("\nDetecting cells...")
    create_detections()
    print("\nClassifying cells...")
    classify_images()
    print("Done.")
    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()
