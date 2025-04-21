import cv2
import json
import math
import numpy as np
import statistics

def load_directions_dict(converted_json_file):
    """
    Expects a JSON file that is already a dictionary, like:
      {
        "x,y": [dx, dy, dz],
        ...
      }
    Returns a Python dict: { "x,y": [dx, dy, dz], ... }
    """
    with open(converted_json_file, 'r') as f:
        directions_dict = json.load(f)
    return directions_dict

def detect_and_match_features(img1, img2, nfeatures=500):
    """
    Detect and match ORB keypoints/descriptors between two images.
    Returns (kp1, kp2, matches), sorted by distance.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # Sort them by distance
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

def angle_between_vectors(vec1, vec2):
    """
    Compute the angle in degrees between two 3D vectors:
      angle = acos( (v1 · v2) / (|v1|*|v2|) )
    """
    v1 = np.array(vec1, dtype=np.float64)
    v2 = np.array(vec2, dtype=np.float64)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return None  # degenerate case
    
    # Normalize to unit vectors
    v1_unit = v1 / norm1
    v2_unit = v2 / norm2
    
    dot_val = np.dot(v1_unit, v2_unit)
    # Numerical safety
    dot_val = np.clip(dot_val, -1.0, 1.0)
    angle_radians = math.acos(dot_val)
    return angle_radians

def compute_average_angle(directions_dict, image_path1, image_path2, needed_matches=5):
    """
    Loads two images, matches keypoints, and for each match tries to:
      1) Round keypoint coords to (x, y)
      2) Lookup "x,y" in directions_dict
      3) If both found, compute angle
    Returns the average angle in degrees (float) or None if not enough matches.
    """
    # 1) Load images
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load one of the images.")
    
    # 2) Attempt different nfeatures to get enough matches
    for nfeatures in [50, 500, 1000, 2000, 5000]:
        print(f"Using {nfeatures} features.")
        kp1, kp2, matches = detect_and_match_features(img1, img2, nfeatures=nfeatures)
        
        angles = []
        for m in matches:
            # Keypoints
            pt1 = kp1[m.queryIdx].pt  # (x1_float, y1_float)
            pt2 = kp2[m.trainIdx].pt  # (x2_float, y2_float)
            
            # Round them to nearest integer
            x1 = int(round(pt1[0]))
            y1 = int(round(pt1[1]))
            x2 = int(round(pt2[0]))
            y2 = int(round(pt2[1]))
            
            key1 = f"{x1},{y1}"
            key2 = f"{x2},{y2}"
            
            # Look up directions
            if key1 in directions_dict and key2 in directions_dict:
                dir1 = directions_dict[key1]
                dir2 = directions_dict[key2]
                angle_deg = angle_between_vectors(dir1, dir2)
                if angle_deg is not None:
                    print(f"Angle between pixel {key1} and pixel {key2} is : {angle_deg} degrees.")
                    angles.append(angle_deg)
        
        if len(angles) >= needed_matches:
            return sum(angles)/len(angles), statistics.median(angles)
    
    # If we exit the loop without returning, we didn't get enough
    return None

if __name__ == "__main__":
    converted_json_file = "pixel_direction_dict_s.json" # Or "pixel_direction_dict.json" for large frodobot
    image_path1 = "1.jpg"
    image_path2 = "2.jpg"
    needed_matches = 10

    directions_dict = load_directions_dict(converted_json_file)
    
    angle_mean, angle_median = compute_average_angle(directions_dict, image_path1, image_path2, needed_matches=needed_matches)
    if angle_mean is not None:
        print(f"Mean and median angle (in degrees) between matched directions: {angle_mean}, {angle_median}")
    else:
        print(f"Could not compute angle; fewer than {needed_matches} matched directions found.")
