import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import os


class BloodCellPreprocessor:
    def __init__(self, width=256, height=256, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
    
    def preprocess(self, input_image):
        """
        Preprocess blood cell images by detecting and cropping the circular cell region
        """
        try:
            if isinstance(input_image, tf.Tensor):
                input_image = input_image.numpy()
            
            if input_image.dtype != np.uint8:
                if input_image.max() <= 1.0:
                    input_image = (input_image * 255).astype(np.uint8)
                else:
                    input_image = input_image.astype(np.uint8)
            
            # Apply Gaussian blur to reduce noise
            image_blur = cv2.GaussianBlur(input_image, (7, 7), 0)
            
            # Convert to HSV color space for better color segmentation
            image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
            
            # Define HSV range for blood cell detection
            min_red = np.array([80, 60, 140])
            max_red = np.array([255, 255, 255])
            
            image_red1 = cv2.inRange(image_blur_hsv, min_red, max_red)
            
            # Find the biggest contour (presumably the blood cell)
            big_contour, mask = self.find_biggest_contour(image_red1)
            
            if big_contour is None:
                # If no contour found, just resize the original image
                return cv2.resize(input_image, (self.width, self.height), 
                                interpolation=self.inter)
            
            # Find minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(big_contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            height, width, channels = input_image.shape
            
            # Calculate cropping boundaries
            y_start = max(0, center[1] - radius)
            y_end = min(height, center[1] + radius)
            x_start = max(0, center[0] - radius)
            x_end = min(width, center[0] + radius)
            
            # Crop the image to the circular region
            cropped_image = input_image[y_start:y_end, x_start:x_end]
            
            # Resize to target dimensions
            return cv2.resize(cropped_image, (self.width, self.height),
                            interpolation=self.inter)
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            # Return resized original image as fallback
            return cv2.resize(input_image, (self.width, self.height),
                            interpolation=self.inter)
    
    def find_biggest_contour(self, image):
        try:
            image = image.copy()
            contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return None, None
            
            biggest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros(image.shape, np.uint8)
            cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
            
            return biggest_contour, mask
        except Exception as e:
            print(f"Contour detection error: {e}")
            return None, None


def preprocess_dataset_offline(source_path, target_path, preprocessor):
    source_path = Path(source_path)
    target_path = Path(target_path)

    print(f"Starting offline preprocessing from {source_path} to {target_path}")

    subset_path = source_path / "TRAIN"
    if subset_path.exists():
        target_subset_path = target_path / "TRAIN"
        target_subset_path.mkdir(parents=True, exist_ok=True)

        print(f"Processing TRAIN subset...")

        # Process each class directory
        for class_dir in subset_path.iterdir():
            if class_dir.is_dir():
                target_class_path = target_subset_path / class_dir.name
                target_class_path.mkdir(exist_ok=True)

                print(f"Processing class: {class_dir.name}")

                image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))

                for i, img_file in enumerate(image_files):
                    try:
                        img = cv2.imread(str(img_file))
                        if img is None or img.size == 0:
                            print(f"Skipping unreadable or empty image: {img_file}")
                            continue

                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        processed_img = preprocessor.preprocess(img)
                        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

                        target_file = target_class_path / img_file.name
                        cv2.imwrite(str(target_file), processed_img)

                        if (i + 1) % 100 == 0:
                            print(f"Processed {i + 1} images")

                    except Exception as e:
                        print(f"Error processing {img_file.name}: {e}")

    print(f"Preprocessing complete! Processed TRAIN dataset saved to {target_path}")