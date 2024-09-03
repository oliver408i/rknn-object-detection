def filter_detections(outputs, threshold=0.3):
    """
    Filters the detections based on a confidence threshold and returns them
    as (xmin, ymin, xmax, ymax, confidence).

    Parameters:
        outputs (np.ndarray): The raw output from the model with shape [1, 5, 8400].
        threshold (float): The confidence threshold to filter detections.

    Returns:
        List of tuples: Each tuple contains (xmin, ymin, xmax, ymax, confidence).
    """
    # Reshape and transpose the output
    reshaped_output = outputs[0].reshape(5, 8400).transpose()

    # Prepare a list to store valid detections
    detections = []

    # Iterate over the reshaped output to filter by confidence
    for row in reshaped_output:
        x_center, y_center, width, height, confidence = row
        
        # Check if the confidence is above the threshold
        if confidence >= threshold:
            # Convert (x_center, y_center, width, height) to (xmin, ymin, xmax, ymax)
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            
            # Append the valid detection to the list
            detections.append((xmin, ymin, xmax, ymax, confidence))

    return detections
