import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate IoU between two boxes (same as before)
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate area of intersection
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = x_overlap * y_overlap

    # Calculate area of union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

# Function to evaluate detections and compute TP, FP, FN
def evaluate_detections(predictions, ground_truths, iou_threshold=0.5):
    # Initialize confusion matrix (for each class)
    classes = ['dog', 'cat']  # Example classes
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for pred in predictions:
        pred_box, pred_class = pred
        matched = False
        for gt in ground_truths:
            gt_box, gt_class = gt
            if pred_class == gt_class and calculate_iou(pred_box, gt_box) >= iou_threshold:
                # Update confusion matrix (True Positive)
                confusion_matrix[classes.index(pred_class), classes.index(pred_class)] += 1
                matched = True
                break
        if not matched:
            # Update confusion matrix (False Positive)
            confusion_matrix[classes.index(pred_class), -1] += 1  # -1 for False Positive (background)

    # Count FN (ground truths without any matching predictions)
    for gt in ground_truths:
        matched = False
        for pred in predictions:
            pred_box, pred_class = pred
            if gt_class == pred_class and calculate_iou(pred_box, gt_box) >= iou_threshold:
                matched = True
                break
        if not matched:
            # Update confusion matrix (False Negative)
            confusion_matrix[classes.index(gt_class), -1] += 1  # -1 for False Negative (background)

    return confusion_matrix, classes

# Example usage
predictions = [((50, 50, 100, 100), 'dog'), ((30, 30, 60, 60), 'cat')]
ground_truths = [((48, 48, 95, 95), 'dog'), ((25, 25, 70, 70), 'cat')]

conf_matrix, classes = evaluate_detections(predictions, ground_truths)

# Visualize the confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=classes + ['Background'], yticklabels=classes + ['Background'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for YOLO Object Detection')
plt.show()
