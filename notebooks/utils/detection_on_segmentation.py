import numpy as np
import cv2
import math

def resize_image(img, target_size=(4000, 6000)):
    original_shape = img.shape[:2]
    if original_shape != target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    return img

def add_mirrored_border(img, top=184, bottom=184, left=148, right=148):
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)

def create_slices_fixed():
    pos_x = np.arange(0, 5785, 482)
    pos_y = np.arange(0, 3857, 482)
    slices = [np.s_[y:y+512, x:x+512] for x in pos_x for y in pos_y]
    return slices

def extract_and_resize_tiles(img, slices, IMG_HEIGHT=128, IMG_WIDTH=128):
    X = np.array([cv2.resize(img[sl], (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
                  for sl in slices], dtype=np.uint8)
    return X

def run_model_on_tiles(model, tiles):
    predictions = model.predict(tiles)
    return predictions

def threshold_predictions(predictions, threshold=0.5):
    predictions = (predictions > threshold).astype(np.uint8) * 255
    return predictions

def reassemble_tiles(predictions, num_splits=13):
    IMG_WIDTH_DEST = 482
    IMG_HEIGHT_DEST = 482

    RESULT_Y = np.zeros((len(predictions), IMG_HEIGHT_DEST, IMG_WIDTH_DEST, 1), dtype=np.uint8)

    for j, x in enumerate(predictions):
        resized = cv2.resize(x.squeeze(), (512, 512), interpolation=cv2.INTER_LINEAR)[15:497, 15:497]
        RESULT_Y[j] = np.expand_dims(resized, axis=-1)

    RESULT_Y_split = np.array_split(RESULT_Y, num_splits)
    reassembled = np.squeeze(np.hstack([np.vstack(group) for group in RESULT_Y_split]))[169:4169, 133:6133]
    return reassembled

def filter_biggest_segmentation_area(segmentation):
    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return segmentation, None
    max_cnt = max(contours, key=cv2.contourArea)
    segmentation_filled = np.zeros_like(segmentation)
    cv2.drawContours(segmentation_filled, [max_cnt], -1, 255, thickness=cv2.FILLED)
    bounding_rect = cv2.boundingRect(max_cnt)
    return segmentation_filled, bounding_rect

def detect_cells(roi):
    try:
        roi_gray = cv2.split(roi)[2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
        roi_gray = clahe.apply(roi_gray)
        roi_gray = cv2.bilateralFilter(roi_gray, 5, 50, 50)

        all_points = []
        for j in range(5, 50, 5):
            points = cv2.HoughCircles(
                roi_gray,
                cv2.HOUGH_GRADIENT,
                dp=2,
                minDist=12,
                param1=145,
                param2=55,
                minRadius=j + 1,
                maxRadius=j + 5,
            )
            if points is not None:
                all_points.extend(points[0][:, :3].astype(np.int32))

        all_points = np.array(all_points)
        if all_points.size == 0:
            best_radius = 33
        else:
            best_radius = np.bincount(all_points[:, 2]).argmax()

        minDist = best_radius * 2 - ((best_radius * 9 / 26) + 75 / 26)
        minRadius = best_radius - max(2, math.floor(best_radius * 0.1))
        maxRadius = best_radius + max(2, math.floor(best_radius * 0.1))

        points = cv2.HoughCircles(
            roi_gray,
            cv2.HOUGH_GRADIENT,
            dp=3,
            minDist=minDist,
            param1=100,
            param2=25,
            minRadius=int(minRadius),
            maxRadius=int(maxRadius),
        )
        if points is not None:
            points = points[0][:, :3].astype(np.int32)
            points = points[(points[:, 0] < roi_gray.shape[1]) & (points[:, 1] < roi_gray.shape[0])]
            return points
        else:
            return np.array([])
    except Exception as e:
        print("Cell detection failed:", e)
        return np.array([])

def remove_detections_outside_segmentation(cells, segmentation):
    if cells.size == 0:
        return cells
    mask = segmentation[cells[:, 1], cells[:, 0]] > 0
    return cells[mask]

def draw_cells(image, cells):
    for c in cells:
        cv2.circle(image, (c[0], c[1]), c[2], (0, 255, 0), 6)
    return image

def detect_cells_on_segmentation(model, img):
    """
    Process an image using the provided segmentation model.

    Parameters:
    - model: The segmentation model.
    - img: The image to process.

    Returns:
    - cells: The detected cells.
    - img_cells: The image with cells drawn.
    """
    img = resize_image(img)
    reflect = add_mirrored_border(img)
    slices = create_slices_fixed()
    X = extract_and_resize_tiles(reflect, slices)
    predictions = run_model_on_tiles(model, X)
    predictions = threshold_predictions(predictions)
    reassembled_segmentation = reassemble_tiles(predictions)
    segmentation_filled, bounding_rect = filter_biggest_segmentation_area(reassembled_segmentation)
    if bounding_rect is None:
        print("No segmentation area found.")
        return None, None
    x, y, w, h = bounding_rect
    roi = img[y:y+h, x:x+w]
    cells = detect_cells(roi)
    if cells.size == 0:
        print("No cells detected.")
        return None, None
    cells[:, 0] += x
    cells[:, 1] += y
    cells = remove_detections_outside_segmentation(cells, segmentation_filled)
    img_cells = draw_cells(img.copy(), cells)
    return cells, img_cells
