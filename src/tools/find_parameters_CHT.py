import click
import sys
sys.path.append("..")
import preprocessing
import cv2
import numpy as np

def nothing(x):
    pass

def create_trackbars():
    # create trackbars for CHT parameters
    cv2.createTrackbar('dp','Find Parameters CHT',3,5,nothing)
    cv2.createTrackbar('minDist','Find Parameters CHT',51,70,nothing)
    cv2.createTrackbar('param1','Find Parameters CHT',100,150,nothing)
    cv2.createTrackbar('param2','Find Parameters CHT',25,100,nothing)

def get_track_bar_value(track_name: str) -> int:
    value = cv2.getTrackbarPos(track_name,'Find Parameters CHT')
    value = value if value else 1
    return int(value)

def draw_cells(image, cells):
    for c in cells:
        image = cv2.circle(image, (c[0], c[1]), (c[2]), (0, 255, 0), 6)
    return image

def detect_cells(image):
    # getting parameters
    dp = get_track_bar_value("dp")
    minDist = get_track_bar_value("minDist")
    param1 = get_track_bar_value("param1")
    param2 = get_track_bar_value("param2")

    # detecting cells usin CHT
    cells = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, 
                             param1=param1, param2=param2, minRadius=31, maxRadius=37)
    if cells is not None:
        cells = cells[0][:,:3]
        cells = cells.astype(np.int32)
    
    return cells

@click.command()
@click.argument('image', type=click.Path(exists=True))
def main(image):
    # Create a black image, a window
    original_img = cv2.imread(image)
    original_preproc_img = preprocessing.pipeline(original_img)
    
    cv2.namedWindow('Find Parameters CHT', 0)
    create_trackbars()
    cv2.imshow("Find Parameters CHT", original_img)

    while(1):        
        k = cv2.waitKey(1) & 0xFF
        
        if k == 27:  # escape
            break
        elif k == 13:  # enter
            print("Detecting cells...")
            cells = detect_cells(original_preproc_img)
            print(f"{len(cells) if cells is not None else 0} detected")
            if cells is not None:
                print("Drawing detected cells...")
                img_with_draw = draw_cells(np.copy(original_img), cells)
                cv2.imshow('Find Parameters CHT', img_with_draw)
            else:
                cv2.imshow('Find Parameters CHT', original_img)
            print("Done.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
