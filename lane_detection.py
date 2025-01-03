import numpy as np
import cv2
import math
import time
from ultralytics import YOLO

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, left_line, right_line, color=[0, 255, 0], thickness=10):
    cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), color, thickness)
    cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), color, thickness)
    return img

def make_points(H, average):
    slope, y_int = average 
    y1 = H
    y2 = int(y1 * (3/5))
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

def lane_detect(image):
    H, W, C = image.shape
    region_of_interest_vertices = [
        (0, H),
        (W / 2, H / 2),
        (W, H),
    ]

    # Convert to grayscale and apply Canny edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)

    # cv2.imshow('Canny', canny_image)

    # Mask out the region of interest
    maksed_image = region_of_interest(canny_image,
                                      np.array([region_of_interest_vertices], np.int32),)

    # cv2.imshow('Masked', maksed_image)

    # Perform Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(
        maksed_image,
        rho=2,
        theta=np.pi / 180,
        threshold=100,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=5
    )

    # Separate left and right lines
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if math.fabs(slope) < 0.5:  # Ignore nearly horizontal lines
                continue
            if slope <= 0:  # Left lane
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:  # Right lane
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    # Average left and right lines
    min_y = int(H * (3 / 5))  # Slightly below the middle of the image
    max_y = H  # Bottom of the image

    if left_line_x and left_line_y:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
    else:
        left_x_start, left_x_end = 0, 0  # Defaults if no lines detected

    if right_line_x and right_line_y:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
    else:
        right_x_start, right_x_end = 0, 0  # Defaults if no lines detected

    return draw_lines(image, [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y])
    
    # # Separate left and right lines
    # left = []
    # right = []
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         if x1 != x2:
    #             slope, y_inter = np.polyfit((x1, x2), (y1, y2), deg=1)
    #             if slope <= 0:  # Left lane
    #                 left.append([slope, y_inter])
    #             else:  # Right lane
    #                 right.append([slope, y_inter])
    
    # left_avg = np.mean(left, axis=0)
    # right_avg = np.mean(right, axis=0)
    # left_line = make_points(H, left_avg)
    # right_line = make_points(H, right_avg)

    # return draw_lines(image, left_line, right_line)

# Function to estimate distance based on bounding box size
def estimate_distance(bbox_width, bbox_height):
    # For simplicity, assume the distance is inversely proportional to the box size
    # This is a basic estimation, you may use camera calibration for more accuracy
    focal_length = 1000  # Example focal length, modify based on camera setup
    known_width = 2.0  # Approximate width of the car (in meters)
    distance = (known_width * focal_length) / bbox_width  # Basic distance estimation
    return distance

def car_detect(model, frame, lane_frame, color=(0, 255, 255), thickness=2):
    # Run YOLO
    yolo_results = model(frame)

    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])

            if model.names[cls] == 'car' and conf >= 0.5:
                # Draw car box
                label = f'{conf:.2f}'
                cv2.rectangle(lane_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(lane_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                # Estimate the distance
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                dist = estimate_distance(bbox_width, bbox_height)

                # Display the estimated distance
                dist_label = f'Dist: {dist:.2f}m'
                cv2.putText(lane_frame, dist_label, (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return lane_frame

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Set frame rate
    target_fps = 30
    frame_time = 1.0 / target_fps

    # Resize to 730p (1280 x 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    model = YOLO("yolo11n.pt")

    # Loop through each frame
    while cap.isOpened():
        # Read the next frame
        ret, frame = cap.read()

        if not ret:
            break

        # Run lane detection
        lane_frame = lane_detect(frame)

        car_detect(model, frame, lane_frame)

        # Display the processed frame
        cv2.imshow('Lane Detection', lane_frame)

        # Limit the frame rate to 30 fps
        time.sleep(frame_time)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # image = cv2.imread('video/test.png')
    # processed_image = lane_detect(image)
    # cv2.imshow(processed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    process_video('video/car.mp4')
