import cv2
import numpy as np
import yaml
import time

# The yaml files contain the camera calibration data, see line 71 for path.
def load_camera_calibration(filename):

    with open (filename, 'r') as file:
        calibration_data = yaml.safe_load(file)

    image_width = calibration_data['image_width']
    image_height = calibration_data['image_height']
    camera_matrix = np.array(calibration_data['camera_matrix']['data']).reshape((3, 3))
    distortion_coefficients = np.array(calibration_data['distortion_coefficients']['data'])
    rectification_matrix = np.array(calibration_data['rectification_matrix']['data']).reshape((3, 3))
    projection_matrix = np.array(calibration_data['projection_matrix']['data']).reshape((3, 4))

    return (image_width, image_height, camera_matrix, distortion_coefficients, rectification_matrix, projection_matrix)

def calibrate_image(image, camera_matrix, distortion_coefficients, new_camera_matrix=None):
    h, w = image.shape[:2]
    if new_camera_matrix is None:
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
    else:
        _, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
    calibrated_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, new_camera_matrix)
    x, y, w, h = roi
    calibrated_image = calibrated_image[y:y+h, x:x+w]
    return calibrated_image

def create_hsv_trackbars(window_name):
    cv2.createTrackbar('H min', window_name, 0, 360, nothing)
    cv2.createTrackbar('H max', window_name, 360, 360, nothing)
    cv2.createTrackbar('S min', window_name, 0, 255, nothing)
    cv2.createTrackbar('S max', window_name, 255, 255, nothing)
    cv2.createTrackbar('V min', window_name, 0, 255, nothing)
    cv2.createTrackbar('V max', window_name, 255, 255, nothing)


def set_hsv_values(window_name, h_min, h_max, s_min, s_max, v_min, v_max):
    cv2.setTrackbarPos('H min', window_name, h_min)
    cv2.setTrackbarPos('H max', window_name, h_max)
    cv2.setTrackbarPos('S min', window_name, s_min)
    cv2.setTrackbarPos('S max', window_name, s_max)
    cv2.setTrackbarPos('V min', window_name, v_min)
    cv2.setTrackbarPos('V max', window_name, v_max)

def get_hsv_values(window_name):
    h_min = cv2.getTrackbarPos('H min', window_name)
    h_max = cv2.getTrackbarPos('H max', window_name)
    s_min = cv2.getTrackbarPos('S min', window_name)
    s_max = cv2.getTrackbarPos('S max', window_name)
    v_min = cv2.getTrackbarPos('V min', window_name)
    v_max = cv2.getTrackbarPos('V max', window_name)
    return (h_min, s_min, v_min), (h_max, s_max, v_max)

# Unused functions for edge detection
def create_edge_detection_trackbars(window_name):
    cv2.createTrackbar('Threshold1', window_name, 399, 500, nothing)
    cv2.createTrackbar('Threshold2', window_name, 400, 500, nothing)

def set_edge_detection_values(window_name, threshold1, threshold2):
    cv2.setTrackbarPos('Threshold1', window_name, threshold1)
    cv2.setTrackbarPos('Threshold2', window_name, threshold2)

def get_edge_detection_values(window_name):
    threshold1 = cv2.getTrackbarPos('Threshold1', window_name)
    threshold2 = cv2.getTrackbarPos('Threshold2', window_name)
    return threshold1, threshold2

# Here we load the calibration data from the yaml file
calibration_file = '/home/krumsvik/Styringssystem/Camera_Calib/calibrationdata/ost.yaml'
(image_width, image_height, camera_matrix,
distortion_coefficients, rectification_matrix,
projection_matrix) = load_camera_calibration(calibration_file)

def add_label(image, text, position=(50, 50)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green color
    thickness = 1
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def apply_gaussian_blur(image, kernel_size):
    # Ensure kernel size is odd and at least 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def nothing(x):
    pass

def draw_chess_grid(image):
    # Define the grid coordinates based on the pixel coordinates
    grid_coordinates = {
        'h8': (513, 600, 412, 480),
        'g8': (508, 590, 348, 411),
        'f8': (508, 600, 281, 347),
        'e8': (505, 608, 223, 280),
        'd8': (505, 595, 167, 222),
        'c8': (505, 585, 110, 166),
        'b8': (505, 585, 52, 109),
        'a8': (505, 585, 0, 51),

        'h7': (447, 512, 413, 480),
        'g7': (445, 510, 349, 412),
        'f7': (445, 507, 280, 348),
        'e7': (445, 504, 220, 279),
        'd7': (441, 504, 167, 219),
        'c7': (441, 504, 110, 166),
        'b7': (438, 504, 52, 109),
        'a7': (435, 504, 0, 51),

        'h6': (384, 446, 413, 479),
        'g6': (383, 444, 348, 412),
        'f6': (381, 444, 284, 347),
        'e6': (378, 444, 220, 283),
        'd6': (377, 440, 166, 219),
        'c6': (375, 440, 111, 165),
        'b6': (375, 437, 55, 110),
        'a6': (373, 434, 0, 54),

        'h5': (316, 383, 413, 480),
        'g5': (316, 382, 350, 412),
        'f5': (313, 380, 287, 349),
        'e5': (312, 377, 227, 286),
        'd5': (311, 376, 168, 226),
        'c5': (311, 374, 111, 167),
        'b5': (310, 374, 57, 110),
        'a5': (309, 372, 0, 56),

        'h4': (246, 315, 414, 480),
        'g4': (246, 315, 351, 415),
        'f4': (245, 312, 286, 350),
        'e4': (245, 311, 226, 285),
        'd4': (245, 310, 168, 225),
        'c4': (245, 310, 111, 167),
        'b4': (246, 309, 58, 110),
        'a4': (247, 308, 0, 57),

        'h3': (180, 245, 420, 480),
        'g3': (180, 245, 351, 419),
        'f3': (177, 244, 291, 350),
        'e3': (180, 244, 228, 290),
        'd3': (181, 244, 171, 227),
        'c3': (182, 244, 114, 170),
        'b3': (182, 245, 58, 113),
        'a3': (185, 246, 0, 57),

        'h2': (113, 179, 420, 480),
        'g2': (112, 179, 353, 419),
        'f2': (109, 176, 292, 352),
        'e2': (112, 179, 230, 291),
        'd2': (116, 180, 172, 229),
        'c2': (120, 181, 115, 171),
        'b2': (121, 181, 56, 114),
        'a2': (120, 184, 0, 55),

        'h1': (20, 112, 421, 480),
        'g1': (23, 111, 353, 420),
        'f1': (25, 108, 293, 352),
        'e1': (32, 111, 231, 292),
        'd1': (37, 115, 170, 230),
        'c1': (42, 119, 116, 169),
        'b1': (47, 120, 54, 115),
        'a1': (45, 119, 0, 53),
    }
    # Draw grid lines based on the provided coordinates
    for key, (x1, x2, y1, y2) in grid_coordinates.items():
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(image, key, (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    return image, grid_coordinates

def find_chess_pieces(image, grid_coordinates, orange_mask, black_mask):
    pieces_positions = {}

    for mask, color in [(orange_mask, 'orange'), (black_mask, 'black')]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                piece_center = (x + w // 2, y + h // 2)

                for key, (x1, x2, y1, y2) in grid_coordinates.items():
                    if x1 <= piece_center[0] <= x2 and y1 <= piece_center[1] <= y2:
                        pieces_positions[key] = (x, y, w, h, color)
                        break  

    return pieces_positions

# If a piece habitates more than one square, the function will determine the most likely position
def determine_piece_position(pieces_positions):
    final_positions = {}
    for key, position in pieces_positions.items():
        final_positions[key] = position
    return final_positions

# Function to compare position to determine if movement has occurred
def compare_piece_positions(old_positions, new_positions, tolerance=25):
    movements = []
    used_new_positions = set()
    print("Old Positions:", old_positions)
    print("New Positions:", new_positions)
    # Setting a tolerance for the movement detection at 15 pixels in x
    def is_within_tolerance(old_pos, new_pos, tol):
        ox, oy, ow, oh, _ = old_pos
        nx, ny, nw, nh, _ = new_pos
        return (abs(ox - nx) <= tol) and (abs(oy - ny) <= tol)

    # Detect pieces that have moved from old positions
    for old_key, old_piece in old_positions.items():
        found = False
        for new_key, new_piece in new_positions.items():
            if new_key in used_new_positions:
                continue
            if old_piece[4] == new_piece[4] and old_key != new_key:
                if old_key not in new_positions and new_key not in old_positions:
                    if not is_within_tolerance(old_piece, new_piece, tolerance):
                        movements.append((old_key, new_key))
                        used_new_positions.add(new_key)
                        found = True
                        break
        if not found:
            movements.append((old_key, None))

    # Detect new pieces that have appeared
    for new_key, new_piece in new_positions.items():
        if new_key not in used_new_positions and new_key not in old_positions:
            for old_key, old_piece in old_positions.items():
                if old_piece[4] == new_piece[4] and old_key not in used_new_positions:
                    if not is_within_tolerance(old_piece, new_piece, tolerance):
                        movements.append((old_key, new_key))
                        used_new_positions.add(new_key)
                        break

    # Detect captures
    for old_key, old_piece in old_positions.items():
        if old_key not in new_positions:
            for new_key, new_piece in new_positions.items():
                if old_piece[4] != new_piece[4] and new_key not in used_new_positions:
                    if not is_within_tolerance(old_piece, new_piece, tolerance):
                        movements.append((old_key, new_key))
                        used_new_positions.add(new_key)
                        break

    # Debug output for detected movements
    print("Detected raw movements:", movements)

    # Filter out invalid movements
    valid_movements = []
    for move in movements:
        from_pos, to_pos = move
        if from_pos and to_pos and from_pos != to_pos:
            if old_positions[from_pos][4] == new_positions[to_pos][4]:
                valid_movements.append(move)
            else:
                print(f"Ignored invalid movement from {from_pos} to {to_pos} (same color swap)")
        elif from_pos and not to_pos:
            valid_movements.append((from_pos, None))
        elif not from_pos and to_pos:
            valid_movements.append((None, to_pos))

    # Debug output for valid movements
    print("Valid movements:", valid_movements)

    return valid_movements

def generate_movement_strings(movements, old_positions, new_positions):
    move_strings = []
    for move in movements:
        from_pos, to_pos = move
        if from_pos and to_pos:
            move_strings.append(f'{from_pos}{to_pos}')
    return move_strings

previous_positions = None

# Open the camera
camera = cv2.VideoCapture(2)
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow('Distorted and Undistorted Frame')
cv2.namedWindow('HSV Orange Mask')
cv2.namedWindow('HSV Black Mask')

cv2.createTrackbar('Kernel size', 'Distorted and Undistorted Frame', 1, 49, nothing)
create_hsv_trackbars('HSV Orange Mask')
create_hsv_trackbars('HSV Black Mask')


cv2.setTrackbarPos('Kernel size', 'Distorted and Undistorted Frame', 5)

cv2.setTrackbarPos('H min', 'HSV Orange Mask', 0)   
cv2.setTrackbarPos('H max', 'HSV Orange Mask', 360)
cv2.setTrackbarPos('S min', 'HSV Orange Mask', 160)
cv2.setTrackbarPos('S max', 'HSV Orange Mask', 255)
cv2.setTrackbarPos('V min', 'HSV Orange Mask', 105)
cv2.setTrackbarPos('V max', 'HSV Orange Mask', 255)

cv2.setTrackbarPos('H min', 'HSV Black Mask', 97)     
cv2.setTrackbarPos('H max', 'HSV Black Mask', 121)
cv2.setTrackbarPos('S min', 'HSV Black Mask', 85)
cv2.setTrackbarPos('S max', 'HSV Black Mask', 217)
cv2.setTrackbarPos('V min', 'HSV Black Mask', 0)
cv2.setTrackbarPos('V max', 'HSV Black Mask', 97)


# Add a delay to allow the camera to focus
time.sleep(1)

# Capture the initial frame and initialize the positions
ret, frame = camera.read()
if ret:
    undistorted_frame = calibrate_image(frame, camera_matrix, distortion_coefficients)
    undistorted_frame_resized = cv2.resize(undistorted_frame, (frame.shape[1], frame.shape[0]))
    blurred_undistorted = apply_gaussian_blur(undistorted_frame_resized, 5)
    hsv_frame = cv2.cvtColor(blurred_undistorted, cv2.COLOR_BGR2HSV)
    orange_lower, orange_upper = get_hsv_values('HSV Orange Mask')
    black_lower, black_upper = get_hsv_values('HSV Black Mask')
    orange_mask = cv2.inRange(hsv_frame, np.array(orange_lower), np.array(orange_upper))
    black_mask = cv2.inRange(hsv_frame, np.array(black_lower), np.array(black_upper))

    kernel = np.ones((2, 2), np.uint8)
    orange_mask = cv2.erode(orange_mask, kernel, iterations=1)
    orange_mask = cv2.dilate(orange_mask, kernel, iterations=1)
    black_mask = cv2.erode(black_mask, kernel, iterations=2)
    black_mask = cv2.dilate(black_mask, kernel, iterations=2)

    labeled_undistorted = add_label(blurred_undistorted.copy(), 'Undistorted')
    undistorted_with_grid, grid_coordinates = draw_chess_grid(labeled_undistorted.copy())
    pieces_positions = find_chess_pieces(undistorted_with_grid, grid_coordinates, orange_mask, black_mask)
    final_positions = determine_piece_position(pieces_positions)
    previous_positions = final_positions
    print("Initial positions captured.")


while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Get current positions of the trackbars
    ksize = cv2.getTrackbarPos('Kernel size', 'Distorted and Undistorted Frame')

    if ksize <= 0:
        ksize = 1  # Kernel size must be at least 1

    # Ensure kernel size is odd
    if ksize % 2 == 0:
        ksize += 1

    # Undistort the frame
    undistorted_frame = calibrate_image(frame, camera_matrix, distortion_coefficients)

    # Resize the undistorted frame to match the original frame's dimensions
    undistorted_frame_resized = cv2.resize(undistorted_frame, (frame.shape[1], frame.shape[0]))

    # Apply Gaussian blur to the undistorted frame
    blurred_undistorted = apply_gaussian_blur(undistorted_frame_resized, ksize)

    # Convert the blurred undistorted frame to HSV
    hsv_frame = cv2.cvtColor(blurred_undistorted, cv2.COLOR_BGR2HSV)

    # Get HSV values for orange and black masks
    orange_lower, orange_upper = get_hsv_values('HSV Orange Mask')
    black_lower, black_upper = get_hsv_values('HSV Black Mask')

    # Create masks for orange and black colors
    orange_mask = cv2.inRange(hsv_frame, np.array(orange_lower), np.array(orange_upper))
    black_mask = cv2.inRange(hsv_frame, np.array(black_lower), np.array(black_upper))

    kernel = np.ones((2, 2), np.uint8)

    orange_mask = cv2.erode(orange_mask, kernel, iterations=2)
    orange_mask = cv2.dilate(orange_mask, kernel, iterations=2)

    black_mask = cv2.erode(black_mask, kernel, iterations=2)
    black_mask = cv2.dilate(black_mask, kernel, iterations=2)

    # Add labels to the distorted and undistorted frames
    labeled_distorted = add_label(frame.copy(), 'Distorted')
    labeled_undistorted = add_label(blurred_undistorted.copy(), 'Undistorted')

    # Draw the chess grid on the undistorted frame
    undistorted_with_grid, grid_coordinates = draw_chess_grid(labeled_undistorted.copy())

    # Find the chess pieces
    pieces_positions = find_chess_pieces(undistorted_with_grid, grid_coordinates, orange_mask, black_mask)
    final_positions = determine_piece_position(pieces_positions)

    # Check for key press to print and send the piece positions
    key = cv2.waitKey(1)
    if key == ord('o'):
        movements = compare_piece_positions(previous_positions, final_positions)
        if movements:
            print("Movements detected:")
            move_strings = generate_movement_strings(movements, previous_positions, final_positions)
            for move_string in move_strings:
                print(move_string)
        else:
            print("No movement detected.")
        previous_positions = final_positions

    # Draw rectangles around detected pieces and label their positions
    for key, (x, y, w, h, color) in final_positions.items():
        cv2.rectangle(undistorted_with_grid, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(undistorted_with_grid, f'{key} ({color})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Stack the labeled frames side by side
    comparison_frame = np.hstack((labeled_distorted, undistorted_with_grid))

    # Display the combined frame
    cv2.imshow('Distorted and Undistorted Frame', comparison_frame)
    cv2.imshow('Undistorted Frame', undistorted_frame_resized)
    cv2.imshow('Undistorted Frame with Grid', undistorted_with_grid)
    cv2.imshow('HSV Orange Mask', orange_mask)
    cv2.imshow('HSV Black Mask', black_mask)

    # Check for key press to exit
    if key == ord('q'):
        break

# Release the camera and close the windows
camera.release()
cv2.destroyAllWindows()
