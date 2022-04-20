# -*- coding: utf-8 -*-

import numpy as np
import os
from tqdm import tqdm
from collections import Counter
from pathlib import PurePath
import imghdr

PATH_IMAGES = os.path.join(*list(PurePath("../original_images/").parts))
PATH_PREDS = os.path.join(*list(PurePath("../annotations/predictions").parts))
PATH_PREDS_CORRECTED = os.path.join(
    *list(PurePath("../annotations/predictions_corrected").parts)
)
OUTPUT_FOLDER = os.path.join(*list(PurePath("../output/spreadsheet").parts))


def cross_plataform_directory():
    global PATH_IMAGES, PATH_PREDS, PATH_PREDS_CORRECTED, OUTPUT_FOLDER
    if "\\" in PATH_IMAGES:
        PATH_IMAGES += "\\"
        PATH_PREDS += "\\"
        PATH_PREDS_CORRECTED += "\\"
        OUTPUT_FOLDER += "\\"
    elif "/" in PATH_IMAGES:
        PATH_IMAGES += "/"
        PATH_PREDS += "/"
        PATH_PREDS_CORRECTED += "/"
        OUTPUT_FOLDER += "/"


def find_image_names():
    l_images = []
    for path, subdirs, files in os.walk(PATH_IMAGES):
        for name in files:
            full_path = os.path.join(path, name)
            if imghdr.what(full_path) is not None:
                l_images.append(full_path.replace(PATH_IMAGES, ""))
    return l_images


def get_images_predictions(folder_im, folder_npy, folder_npy_correct):
    images = find_image_names()  # os.listdir(folder_im)

    im_to_npy = lambda x: os.path.join(
        os.path.join(*PurePath(x).parts[:-1]),
        PurePath(x).parts[-1].split(".")[:1][0] + ".npy",
    )

    files_path = sorted(
        [
            [
                i,
                os.path.join(folder_im, i),
                os.path.join(folder_npy, im_to_npy(i)),
                os.path.join(folder_npy_correct, im_to_npy(i)),
            ]
            for i in images
        ]
    )

    # filter if the image exists and has an annotation
    imgs_with_det = list(
        filter(lambda x: os.path.isfile(x[1]) and os.path.isfile(x[2]), files_path)
    )

    pred_files = []
    rel_paths = []

    for i in imgs_with_det:
        index = -1 if os.path.isfile(i[-1]) else -2
        pth_to_npy = i[index]
        pref_path = PATH_PREDS_CORRECTED if index == -1 else PATH_PREDS
        rel_pth_pred = pth_to_npy.replace(pref_path, "")
        rel_pth_pred = os.path.join(*PurePath(rel_pth_pred).parts[:-1])

        pred_files.append([pth_to_npy, rel_pth_pred])
        rel_paths.append(rel_pth_pred)

    return (pred_files, list(set(rel_paths)))


def get_qtd_by_class(points, labels):
    points_filtered = points[points[:, 6] == 1, 4]
    sum_predictions = Counter(points_filtered)
    return [
        *[str(sum_predictions[i]) for i, j in enumerate(labels)],
        str(len(points_filtered)),
    ]


def get_header(labels):
    return "Img Name," + ",".join([i for i in labels]) + ",Total\n"


def create_folder(path):
    path = os.path.join(*PurePath(path).parts[:-1])
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    cross_plataform_directory()
    pred_files, rel_paths = get_images_predictions(
        PATH_IMAGES, PATH_PREDS, PATH_PREDS_CORRECTED
    )

    labels = ["Capped", "Eggs", "Honey", "Larva", "Nectar", "Other", "Pollen"]

    with tqdm(total=len(rel_paths)) as t:
        for ph in rel_paths:
            npy_in_pth = [i[0] for i in pred_files if i[1] == ph]
            csv_name = PurePath(npy_in_pth[0]).parts[-2] + ".csv"
            csv_path = os.path.join(OUTPUT_FOLDER, ph, csv_name)

            create_folder(csv_path)

            with open(csv_path, "w") as file:
                file.write(get_header(labels))

                for i in npy_in_pth:
                    frame_name = PurePath(i).stem
                    points = np.load(i)[2].astype(np.int32)
                    qtd = get_qtd_by_class(points, labels)
                    if qtd:
                        file.write(frame_name + "," + ",".join(qtd) + "\n")

            t.update(1)

    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()
