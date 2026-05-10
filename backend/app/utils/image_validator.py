import cv2
import numpy as np

def check_blur(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return {
        "is_blurry": bool(variance < 100),
        "blur_score": float(round(variance, 2))
    }

def check_brightness(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)

    return {
        "too_dark": bool(brightness < 50),
        "brightness_score": float(round(brightness, 2))
    }