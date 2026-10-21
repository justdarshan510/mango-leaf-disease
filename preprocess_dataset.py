import os
import cv2
import numpy as np

SRC_DIR = r"dataset\Augmented Mango dataset"
DST_DIR = r"dataset_cropped"

def get_leaf_bounding_box_and_crop(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    combined = cv2.bitwise_or(edges, thresh2)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, img # return original if no contour found
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    img_h, img_w = img.shape[:2]
    if (w * h) < (0.05 * img_w * img_h):
        return False, img # return original if box too small
        
    # Pad slightly
    pad = 10
    y1 = max(0, y - pad)
    y2 = min(img_h, y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(img_w, x + w + pad)
    
    cropped_img = img[y1:y2, x1:x2]
    return True, cropped_img

def process_dataset():
    if not os.path.exists(SRC_DIR):
        print(f"Source directory {SRC_DIR} not found.")
        return

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)
        
    for class_name in os.listdir(SRC_DIR):
        src_class_path = os.path.join(SRC_DIR, class_name)
        if not os.path.isdir(src_class_path):
            continue
            
        dst_class_path = os.path.join(DST_DIR, class_name)
        if not os.path.exists(dst_class_path):
            os.makedirs(dst_class_path)
            
        images = os.listdir(src_class_path)
        print(f"Processing {class_name}... ({len(images)} images)")
        
        success_count = 0
        for img_name in images:
            src_img_path = os.path.join(src_class_path, img_name)
            dst_img_path = os.path.join(dst_class_path, img_name)
            
            success, cropped = get_leaf_bounding_box_and_crop(src_img_path)
            if cropped is not None:
                cv2.imwrite(dst_img_path, cropped)
                if success: success_count += 1
            else:
                print(f"  Failed to read {src_img_path}")
                
        print(f"  Cropped {success_count}/{len(images)} images.")
                
if __name__ == '__main__':
    print("Starting dataset preprocessing...")
    process_dataset()
    print("Preprocessing complete!")
