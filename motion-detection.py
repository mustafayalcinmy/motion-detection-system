import cv2
import numpy as np
from collections import deque

def subtract_images(background, frame, threshold_value):
    diff = cv2.absdiff(background, frame)
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    return diff, thresh

def update_memory(memory, frame, memory_size):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    memory.append(gray_frame)
    if len(memory) < memory_size:
        raise ValueError("Memory not full. Collect more frames to compute background.")

def compute_background(memory):
    return np.median(np.array(memory), axis=0).astype(np.uint8)

def process_frame(frame, memory, threshold_value):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = compute_background(memory)
    _, thresh = subtract_images(background, gray_frame, threshold_value)

    dilated = cv2.dilate(thresh, None, iterations=7)
    eroded = cv2.erode(dilated, None, iterations=9)

    return eroded

def find_contours(binary_image, min_area):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []

    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            hull = cv2.convexHull(approx)
            x, y, w, h = cv2.boundingRect(hull)
            rectangles.append((x, y, w, h))

    return rectangles

def rectangles_overlap(rect1, rect2, buffer_size, overlap_threshold):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    x1 -= buffer_size
    y1 -= buffer_size
    w1 += 2 * buffer_size
    h1 += 2 * buffer_size

    x2 -= buffer_size
    y2 -= buffer_size
    w2 += 2 * buffer_size
    h2 += 2 * buffer_size

    overlap_width = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    overlap_height = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = overlap_width * overlap_height

    area1 = w1 * h1
    area2 = w2 * h2

    overlap_ratio = overlap_area / min(area1, area2)
    return overlap_ratio > overlap_threshold

def merge_two_rectangles(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y

    return (x, y, w, h)

def merge_overlapping_rectangles(rectangles, buffer_size, overlap_threshold):
    merged = True

    while merged:
        merged = False
        for i, rect1 in enumerate(rectangles):
            for j, rect2 in enumerate(rectangles):
                if i >= j:
                    continue

                if rectangles_overlap(rect1, rect2, buffer_size, overlap_threshold):
                    new_rect = merge_two_rectangles(rect1, rect2)
                    rectangles.pop(j)
                    rectangles.pop(i)
                    rectangles.append(new_rect)
                    merged = True
                    break
            if merged:
                break

    return rectangles

def main_motion_detection(threshold_value=30, min_area=500, memory_size=20, overlap_threshold=0.001, buffer_size=30):
    cap = cv2.VideoCapture(0)
    memory = deque(maxlen=memory_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (720, 480))
        try:
            update_memory(memory, frame, memory_size)
            binary_image = process_frame(frame, memory, threshold_value)
            rectangles = find_contours(binary_image, min_area)
            merged_rectangles = merge_overlapping_rectangles(rectangles, buffer_size, overlap_threshold)

            for x, y, w, h in merged_rectangles:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Motion Detection", frame)
            cv2.imshow("Binary Image", binary_image)

        except ValueError:
            cv2.putText(frame, "Collecting frames...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Motion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_motion_detection()
