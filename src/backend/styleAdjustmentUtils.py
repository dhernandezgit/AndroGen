import re
import numpy as np
import cv2

def process_annotation(annotation):
    if annotation["boxes"]:
        box = annotation["boxes"][0]
        crop = annotation["image"][
            box["ymin"]:box["ymax"],
            box["xmin"]:box["xmax"]]
        print(box)
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Detect edges to identify the central item
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in the image.")
        
        # Find the largest contour, assuming it is the central item
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create masks for central item, border, and outside border regions
        mask_central = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_central, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        mask_border = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_border, [largest_contour], -1, 255, thickness=5)  # Border thickness
        
        mask_outside = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_outside, [largest_contour], -1, 255, thickness=15)
        
        # Extract average colors from each region
        def calculate_median_color(image, mask):
            masked_pixels = image[mask > 0]
            if masked_pixels.size == 0:
                return (0, 0, 0)  # Default to black if no pixels are in the region
            return tuple(map(int, np.median(masked_pixels, axis=0)))
        
        central_color = calculate_median_color(crop, mask_central)
        border_color = calculate_median_color(crop, mask_border)
        outside_color = calculate_median_color(crop, mask_outside)

        return ["rgba({}, {}, {}, 1)".format(*central_color),
                "rgba({}, {}, {}, 1)".format(*central_color),
                "rgba({}, {}, {}, 1)".format(*central_color),
                "rgba({}, {}, {}, 1)".format(*central_color),
                "rgba({}, {}, {}, 1)".format(*central_color),
                "rgba({}, {}, {}, 1)".format(*border_color),
                "rgba({}, {}, {}, 1)".format(*outside_color)]

def get_color(annotation):
    if annotation["boxes"]:
        box = annotation["boxes"][0]
        crop = annotation["image"][
            box["ymin"]:box["ymax"],
            box["xmin"]:box["xmax"]]
        mask = np.ones(crop.shape[:2], dtype=np.uint8)

        def calculate_median_color(image, mask):
            return tuple(map(int, np.median(image[mask > 0], axis=0)))
        
        color = calculate_median_color(crop, mask)
        return "rgba({}, {}, {}, 1)".format(*color)

def process_color(color):
    # Check if color is in RGBA format
    if color.startswith("rgba"):
        # Parse RGBA string
        rgba_values = re.findall(r"[\d.]+", color)
        r, g, b, _ = map(float, rgba_values)  # Ignore alpha
        return [r, g, b]
    else:
        # Parse Hex string
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return [r, g, b]