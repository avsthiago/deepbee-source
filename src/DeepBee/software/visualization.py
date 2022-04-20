#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:16:22 2018

@author: avsthiago
"""
import os
import cv2
import numpy as np
import datetime
import imghdr

# RESOLUTIONS [0] 1920X1080 [1] 1280X720
RESOLUTION = 0
CONFIG_FILE = 'config.txt'

IMAGES_PATH = '../original_images/'
DET_PRED_PATH = '../annotations/predictions'
MOVING_DIRECTION = 'F'
OUT_ANNOT = '../annotations/predictions_corrected'
OUT_IMGS = '../output/labeled_images'
LEFT_BAR_SIZE = None
SIZE_WINDOW_COMB = None
X_POS_WINDOW_COMB = None
OPTION = 32
RECTANGLE = None
COLORS = [(255, 0, 0), (0, 255, 255), (0, 0, 128), (255, 0, 255), (0, 255, 0),(255, 255, 100), (0, 0, 255)]

LABELS = ["Capped",  "Eggs", "Honey", "Larva", "Nectar", "Other", "Pollen"]

CLASSES_KEY = [49, 50, 51, 52, 53, 54, 55]
REFRESH = False
REFRESH_RECTANGLE = False
RECTANGLE_MODE = False
MOUSE_PRESSED = False
LABELING_MODE = False
ADDING_MODE = False
REMOVING_MODE = False
MOVING_MODE = True
VIEW_MODE = False
AVG_CELL_SIZE = 30
IX, IY = 0, 0
MIN_DIST = 0
LAST_SAVING = ''
MIN_CONFIDENCE = 99950

OLD_GAMMA = 0
OLD_BRIGHTNESS = 0
OLD_CONTRAST = 0
GAMMA = 2
BRIGHTNESS = 0
CONTRAST = 0
IM_NAME = ""


def load_configs():
    global RESOLUTION
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            lines = file.readlines()
            # load RESOLUTION
            try:
                mx_sam = [i for i in lines if 'RESOLUTION' in i][0]
                RESOLUTION = int(mx_sam.split(':')[-1].strip())
            except:
                print('RESOLUTION not found in the config.txt file')
            # load batch


def set_window_size():
    global LEFT_BAR_SIZE, X_POS_WINDOW_COMB, SIZE_WINDOW_COMB 
    if RESOLUTION == 0:
        LEFT_BAR_SIZE = (1000, 300)
        SIZE_WINDOW_COMB = (1500, 1000)
        X_POS_WINDOW_COMB = 300
    elif RESOLUTION == 1:
        LEFT_BAR_SIZE = (1000, 300)
        SIZE_WINDOW_COMB = (980, 1000)
        X_POS_WINDOW_COMB = 300

  
def draw_circles(img):
    img2 = np.copy(img)
    for c in POINTS:
        if c[6]:
            cv2.circle(img2, (c[0], c[1]), c[2], COLORS[c[4]], 5)
            if not c[5]:
                cv2.circle(img2, (c[0], c[1]), int(max(.1*AVG_CELL_SIZE+3, 3)), 
                           (0,0, 255), -1)
    return img2


def draw_rectangle(img):
    return cv2.rectangle(np.copy(img), *RECTANGLE, (0, 255,0),10)


def count_cells_by_class(cl):
    cond = np.where(np.logical_and(POINTS[:, 4] == cl, POINTS[:, 6] == 1))
    return str(len(POINTS[cond]))


def generate_left_bar_image(saving=False):
    left_panel = np.zeros((LEFT_BAR_SIZE[0], LEFT_BAR_SIZE[1], 3),
                          dtype=np.uint8)
    for i, color_lb in enumerate(zip(COLORS, LABELS)):
        color, label = color_lb
        if saving:
            chosen = ""
        else:
            chosen = "[X]" if OPTION == i + 49 else "[ ]"  # 49 = key-1
        cv2.putText(left_panel, " ".join([chosen , str(i+1), ".", label, count_cells_by_class(i)]),
                    (15, 45*(i+1)), cv2.FONT_HERSHEY_DUPLEX, .8, color, 2)

    cv2.putText(left_panel, "Saved:",
                (15, 45*20), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 255, 255), 2)
    cv2.putText(left_panel, LAST_SAVING,
                (15, 45*21), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 255, 255), 2)

    if not saving:
        chosen = "[X]" if OPTION == 32 else "[ ]"
        cv2.putText(left_panel, " ".join([chosen , 'Space .', 'Move']),
                        (15, int(45*8.5)), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 255, 255), 2)
        chosen = "[X]" if OPTION == 114 else "[ ]"
        key = 'Enter . Finish Sel.' if RECTANGLE_MODE else 'R . Select Area'
        cv2.putText(left_panel, " ".join([chosen , key]),
                        (15, int(45*9.5)), cv2.FONT_HERSHEY_DUPLEX, .7, (255, 255, 255), 2)

        cv2.putText(left_panel, "[V]. Toggle Detect.",
                    (15, 45*11), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 255, 255), 2)
        cv2.putText(left_panel, "[A]. Add Cell",
                    (15, 45*12), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 255, 255), 2)
        cv2.putText(left_panel, "[D]. Del. Cell",
                    (15, 45*13), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 255, 255), 2)
        cv2.putText(left_panel, "[S]. Save",
                    (15, 45*14), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 255, 255), 2)
        cv2.putText(left_panel, "[N]. Next",
                    (15, 45*15), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 255, 255), 2)
        cv2.putText(left_panel, "[P]. Previous",
                    (15, 45*16), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 255, 255), 2)
        cv2.putText(left_panel, "[BS]. Reset",
                    (15, 45*17), cv2.FONT_HERSHEY_DUPLEX, .8, (0, 0, 255), 2)
        cv2.putText(left_panel, "[ESC]. Quit",
                    (15, 45*18), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 255, 255), 2)

        cv2.namedWindow('Options', 0)
        cv2.resizeWindow('Options', LEFT_BAR_SIZE[1], LEFT_BAR_SIZE[0])
        cv2.moveWindow('Options', 0, 0)
        cv2.imshow('Options', left_panel)

    return left_panel


def find_image_names():
    l_images = []
    for path, subdirs, files in os.walk(IMAGES_PATH):
        for name in files:
            full_path = os.path.join(path, name)
            if imghdr.what(full_path) is not None:
                l_images.append(full_path.replace(IMAGES_PATH, ''))  
    return l_images


def get_images_predictions(folder_im, folder_npy, folder_npy_correct):
    images = find_image_names()
    im_to_npy = lambda x: os.path.join("/".join(x.split('/')[:-1]) ,
                        ".".join(x.split('/')[-1].split('.')[:1]) + '.npy')
    
    files_path = sorted([[i, os.path.join(folder_im, i),
                          os.path.join(folder_npy, im_to_npy(i)),
                          os.path.join(folder_npy_correct, im_to_npy(i))]
                        for i in images])

    return list(filter(lambda x: os.path.isfile(x[1])
                                 and os.path.isfile(x[2]), files_path))


def next_image(direction, len_images, index):
    if direction == 'F':
        if index + 1 >= len_images:
            return 0
        return index + 1
    if index - 1 <= -1:
        return len_images - 1
    return index - 1


def change_mode(mode):
    global MOVING_MODE, RECTANGLE_MODE, LABELING_MODE, MOUSE_PRESSED, \
           ADDING_MODE, REMOVING_MODE, VIEW_MODE

    if mode == 'VIEW_MODE':
        VIEW_MODE = True
        RECTANGLE_MODE = False
        MOVING_MODE = True
        ADDING_MODE = False
        REMOVING_MODE = False
        LABELING_MODE = False
    else:
        VIEW_MODE = False
        if mode == 'RECTANGLE_MODE':
            RECTANGLE_MODE = True
            MOVING_MODE = False
            ADDING_MODE = False
            REMOVING_MODE = False
        else:
            if RECTANGLE_MODE:
                MOUSE_PRESSED = False
                finish_roi_selection()
    
            RECTANGLE_MODE = False
    
            if mode == 'MOVING_MODE':
                LABELING_MODE = False
                MOVING_MODE = True
                ADDING_MODE = False
                REMOVING_MODE = False
            elif mode == 'LABELING_MODE':
                LABELING_MODE = True
                MOVING_MODE = False
                ADDING_MODE = False
                REMOVING_MODE = False
            elif mode == 'ADDING_MODE':
                LABELING_MODE = False
                MOVING_MODE = False
                ADDING_MODE = True
                REMOVING_MODE = False
            elif mode == 'REMOVING_MODE':
                LABELING_MODE = False
                MOVING_MODE = False
                ADDING_MODE = False
                REMOVING_MODE = True
            
    generate_left_bar_image()

def load_data(list_files):
    if os.path.isfile(list_files[-1]):
        data_loaded = np.load(list_files[-1], allow_pickle=True)
    else:
        data_loaded = np.load(list_files[-2], allow_pickle=True)
    
    RECTANGLE, LAST_SAVING, POINTS = tuple(data_loaded)
    POINTS[:, 7] = (POINTS[:, 7] * 100000).astype(np.int32)
    POINTS = POINTS.astype(np.int32)
    return (RECTANGLE, LAST_SAVING, POINTS)

def process_entries(img_det):
    global OPTION, POINTS, REFRESH, MOVING_MODE, MOVING_DIRECTION, \
           RECTANGLE_MODE, MIN_DIST, LAST_SAVING, RECTANGLE, \
           REFRESH_RECTANGLE, AVG_CELL_SIZE, VIEW_MODE

    img = cv2.imread(img_det[1])
    img_bk = np.copy(img)

    RECTANGLE, LAST_SAVING, POINTS = load_data(img_det)
    MIN_DIST = np.average(POINTS[:, 2])
    AVG_CELL_SIZE = int(np.bincount(POINTS[:, 2]).argmax())

    generate_left_bar_image()

    img = process_trackbar(img)
    img = draw_circles(img)

    cv2.imshow(img_det[0], draw_rectangle(img))
    last_key = 1
    while True:
        k = cv2.waitKey(1)  # & 0xFF
        if k != last_key:
            if k == 27: # Esc close stop
                return False
            if k in CLASSES_KEY:  # Classes
                OPTION = k
                change_mode('LABELING_MODE')
            elif k == 32:  # Space Moving image
                OPTION = 32
                change_mode('MOVING_MODE')
            elif k == 97:  # A add cell
                change_mode('ADDING_MODE')
            elif k == 100:  # D delete cell
                change_mode('REMOVING_MODE')
            elif k == 114:  # R key select ROI
                OPTION = 114
                change_mode('RECTANGLE_MODE')
            elif k == 118:  # v key view cells without label
                if VIEW_MODE:
                    OPTION = 32
                    change_mode('MOVING_MODE')
                else:
                    OPTION = 118
                    change_mode('VIEW_MODE')
                REFRESH = True
            elif k == 13:  # Enter key Finish ROI selection
                if RECTANGLE_MODE:
                    OPTION = 32
                    change_mode('MOVING_MODE')
                    finish_roi_selection()
                    generate_left_bar_image()
            elif k == 110:  # N key to next image
                MOVING_DIRECTION = 'F'
                change_mode('MOVING_MODE')
                return True
            elif k == 112:  # L key to previous\ image
                MOVING_DIRECTION = 'B'
                change_mode('MOVING_MODE')
                return True
            elif k == 115:
                save(draw_rectangle(img), img_det[0], img_det[-1])
            elif k == 8:
                if os.path.isfile(img_det[-1]):
                    os.remove(img_det[-1])
                process_entries(img_det)

            last_key = k

        if REFRESH:
            img = process_trackbar(np.copy(img_bk))
            REFRESH = False
            if not VIEW_MODE:
                img = draw_circles(img)
                cv2.imshow(img_det[0], draw_rectangle(img))
            else:
                cv2.imshow(img_det[0], img)
        if REFRESH_RECTANGLE:
            img = np.copy(img_bk)
            #img = adjust_gamma(img, GAMMA/2 if GAMMA > 0 else 0.1)
            img = process_trackbar(img)
            img = draw_circles(img)
            cv2.imshow(img_det[0], draw_rectangle(img))
            REFRESH_RECTANGLE = False


def finish_roi_selection():
    global REFRESH_RECTANGLE, POINTS
    (xmin, ymin), (xmax, ymax) = RECTANGLE
    POINTS[:, 6] = 0
    cond = np.where(np.logical_and(POINTS[:,0] >= xmin, POINTS[:,0] <= xmax) &
                    np.logical_and(POINTS[:,1] >= ymin, POINTS[:,1] <= ymax))
    POINTS[cond, 6] = 1
    REFRESH_RECTANGLE = True

def get_selected_class():
    return OPTION - 49

def hover_cell(x, y):
    global MIN_DIST, POINTS, REFRESH

    point = np.array([x, y])
    min_dist_index = np.sum(np.square(np.abs(point-POINTS[:,:2])),1).argmin()
    min_dist = np.sqrt(np.sum(np.square(np.abs(point-POINTS[min_dist_index,:2])),0))

    if min_dist <= MIN_DIST and POINTS[min_dist_index, 6]:
        if POINTS[min_dist_index, 4] != get_selected_class():
            POINTS[min_dist_index, 4] = get_selected_class()
            POINTS[min_dist_index, 5] = 1# if POINTS[min_dist_index, 5] == 0 else 0
        else:
            POINTS[min_dist_index, 5] = 0 if POINTS[min_dist_index, 5] == 1 else 1
        
        REFRESH = True
        
def callback_rectangle(lbt_down, mouse_move, lbt_up, x, y):
    global IX, IY,  RECTANGLE, MOUSE_PRESSED, REFRESH

    if lbt_down:
        MOUSE_PRESSED = True
        IX, IY = x, y
        RECTANGLE = ((IX, IY), (x, y))
    elif mouse_move:
        if MOUSE_PRESSED:
            RECTANGLE = ((IX, IY), (x, y))
            REFRESH = True
    elif lbt_up:
        MOUSE_PRESSED = False
        RECTANGLE = ((IX, IY), (x, y))
        REFRESH = True


def callback_labeling(lbt_down, mouse_move, lbt_up, x, y):
    global MOUSE_PRESSED, REFRESH
    if lbt_down or (mouse_move and MOUSE_PRESSED):
        MOUSE_PRESSED = True
        hover_cell(x, y)
    elif lbt_up:
        MOUSE_PRESSED = False
        generate_left_bar_image()


def callback_adding(lbt_down, mouse_move, lbt_up, x, y):
    global POINTS, REFRESH
    if lbt_down:
        (xmin, ymin), (xmax, ymax) = RECTANGLE
        if xmin <= x <= xmax and ymin <= y <= ymax:
            cl = get_selected_class() if OPTION in CLASSES_KEY else 1  # other
            # stands for in ROI
            new_cell = np.array([x, y, AVG_CELL_SIZE, cl, cl, 1, 1, 1])
            POINTS = np.vstack((POINTS, new_cell))
    elif lbt_up:
        generate_left_bar_image()
        REFRESH = True


def callback_remove(lbt_down, mouse_move, lbt_up, x, y):
    global POINTS, REFRESH
    if lbt_down:
        (xmin, ymin), (xmax, ymax) = RECTANGLE
        if xmin <= x <= xmax and ymin <= y <= ymax:
            point = np.array([x, y])
            min_dist_index = np.sum(np.square(np.abs(
                                    point-POINTS[:, :2])), 1).argmin()
            POINTS = np.delete(POINTS, min_dist_index, 0)
    elif lbt_up:
        generate_left_bar_image()
        REFRESH = True


def annotate_cells(event, x, y, flags, param):
    global MOUSE_PRESSED, RECTANGLE_MODE, MOVING_MODE, LABELING_MODE, \
           ADDING_MODE, REMOVING_MODE, VIEW_MODE

    if RECTANGLE_MODE:
        MOVING_MODE = False
        LABELING_MODE = False
        ADDING_MODE = False
        REMOVING_MODE = False
        callback_rectangle(event == cv2.EVENT_LBUTTONDOWN,
                           event == cv2.EVENT_MOUSEMOVE,
                           event == cv2.EVENT_LBUTTONUP, x, y)
    elif LABELING_MODE:
        MOVING_MODE = False
        RECTANGLE_MODE = False
        ADDING_MODE = False
        REMOVING_MODE = False
        callback_labeling(event == cv2.EVENT_LBUTTONDOWN,
                          event == cv2.EVENT_MOUSEMOVE,
                          event == cv2.EVENT_LBUTTONUP, x, y)
    elif ADDING_MODE:
        MOVING_MODE = False
        RECTANGLE_MODE = False
        LABELING_MODE = False
        REMOVING_MODE = False
        callback_adding(event == cv2.EVENT_LBUTTONDOWN,
                        event == cv2.EVENT_MOUSEMOVE,
                        event == cv2.EVENT_LBUTTONUP, x, y)
    elif REMOVING_MODE:
        MOVING_MODE = False
        RECTANGLE_MODE = False
        LABELING_MODE = False
        ADDING_MODE = False
        callback_remove(event == cv2.EVENT_LBUTTONDOWN,
                        event == cv2.EVENT_MOUSEMOVE,
                        event == cv2.EVENT_LBUTTONUP, x, y)
    elif MOVING_MODE or VIEW_MODE:
        pass


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def process_trackbar(img):
    global REFRESH, IM_NAME, OLD_GAMMA, OLD_BRIGHTNESS, OLD_CONTRAST, REFRESH_RECTANGLE
    if GAMMA != OLD_GAMMA or REFRESH or REFRESH_RECTANGLE:
        gamma = GAMMA/2
        img = adjust_gamma(img, 0.1 if gamma == 0 else gamma)
    
    OLD_GAMMA = GAMMA
    return img


def trackbar(x):
    global REFRESH, GAMMA,  BRIGHTNESS, CONTRAST, IM_NAME, OLD_GAMMA, OLD_BRIGHTNESS, OLD_CONTRAST
    GAMMA = cv2.getTrackbarPos("Gamma", IM_NAME)
    REFRESH = True


def save(img, im_name, out_npy):
    global LEFT_BAR_SIZE, LAST_SAVING
    old_size = LEFT_BAR_SIZE
    LAST_SAVING = datetime.datetime.now().strftime('%d/%m/%y %H:%M:%S')
    out_path = os.path.join(OUT_IMGS, im_name)

    LEFT_BAR_SIZE = (img.shape[0], LEFT_BAR_SIZE[1])
    left_bar = generate_left_bar_image(True)
    cv2.imwrite(out_path, np.hstack((left_bar, img)))

    LEFT_BAR_SIZE = old_size
    
    create_folder(out_npy)
    
    np.save(out_npy, np.array([RECTANGLE, LAST_SAVING, POINTS]))
    generate_left_bar_image()


def create_folder(path):
    path = "/".join(path.split('/')[:-1])
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    global MOVING_DIRECTION, OPTION, POINTS, IM_NAME
    list_im_det = get_images_predictions(IMAGES_PATH, DET_PRED_PATH, OUT_ANNOT)
    set_window_size()
    index = 0

    while True:
        i = list_im_det[index]

        OPTION = 32
        POINTS = np.array([])
        IM_NAME = i[0]
        
        cv2.namedWindow(i[0], 0)
        cv2.createTrackbar("Gamma", i[0], 2,10,trackbar)
        cv2.setMouseCallback(i[0], annotate_cells)
        cv2.resizeWindow(i[0], *(SIZE_WINDOW_COMB))
        cv2.moveWindow(i[0], X_POS_WINDOW_COMB, 0)
        
        if not process_entries(i):
            break

        index = next_image(MOVING_DIRECTION, len(list_im_det), index)

        cv2.destroyAllWindows()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


