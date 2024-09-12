import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src import preprocessing
import cv2
import numpy as np
from typing import Optional, Tuple

def detect_cells(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect cells in the given image using Hough Circle Transform.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        Optional[np.ndarray]: Detected cells as (x, y, radius) or None if no cells detected.
    """
    image = preprocessing.pipeline(image)
    
    all_cells = []
    for radius in range(5, 50, 5):
        cells = cv2.HoughCircles(
            image, 
            cv2.HOUGH_GRADIENT, 
            dp=2, 
            minDist=12, 
            param1=145, 
            param2=55, 
            minRadius=radius + 1, 
            maxRadius=radius + 5
        )
        
        if cells is not None:
            all_cells.extend(cells[0][:, :3].astype(np.int32))
    
    if not all_cells:
        best_radius = 33
    else:
        best_radius = np.bincount([cell[2] for cell in all_cells]).argmax()

    min_dist = best_radius * 2 - ((best_radius * 9/26) + 75/26)
    min_radius = best_radius - max(2, int(best_radius * 0.1))
    max_radius = best_radius + max(2, int(best_radius * 0.1))
    
    cells = cv2.HoughCircles(
        image, 
        cv2.HOUGH_GRADIENT, 
        dp=3, 
        minDist=min_dist,  
        param1=100, 
        param2=25, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )
    
    return cells[0][:, :3].astype(np.int32) if cells is not None else None

def main():
    # Example usage
    image = cv2.imread('path_to_your_image.jpg', 0)  # Read as grayscale
    detected_cells = detect_cells(image)
    
    if detected_cells is not None:
        print(f"Detected {len(detected_cells)} cells")
    else:
        print("No cells detected")

if __name__ == "__main__":
    main()