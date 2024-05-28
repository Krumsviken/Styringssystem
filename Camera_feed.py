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
        'h8': (513, 580, 402, 470),
        'g8': (508, 590, 340, 401),
        'f8': (508, 600, 281, 339),
        'e8': (505, 608, 223, 279),
        'd8': (505, 590, 167, 222),
        'c8': (500, 585, 115, 166),
        'b8': (500, 585, 60, 114),
        'a8': (495, 585, 0, 59),

        'h7': (447, 512, 400, 472),
        'g7': (445, 510, 340, 400),
        'f7': (445, 507, 280, 339),
        'e7': (445, 504, 220, 279),
        'd7': (445, 504, 160, 219),
        'c7': (441, 499, 110, 159),
        'b7': (441, 499, 57, 109),
        'a7': (438, 494, 0, 56),

        'h6': (384, 446, 405, 479),
        'g6': (383, 444, 340, 404),
        'f6': (383, 444, 280, 339),
        'e6': (383, 444, 220, 279),
        'd6': (383, 444, 160, 219),
        'c6': (381, 440, 102, 159),
        'b6': (380, 440, 55, 101),
        'a6': (380, 437, 0, 54),

        'h5': (316, 383, 405, 480),
        'g5': (316, 382, 340, 404),
        'f5': (316, 382, 280, 339),
        'e5': (317, 382, 220, 279),
        'd5': (318, 382, 159, 219),
        'c5': (318, 380, 101, 158),
        'b5': (318, 379, 50, 100),
        'a5': (318, 379, 0, 49),

        'h4': (250, 315, 405, 480),
        'g4': (250, 315, 340, 404),
        'f4': (250, 315, 282, 339),
        'e4': (250, 316, 218, 279),
        'd4': (253, 317, 160, 219),
        'c4': (255, 317, 100, 159),
        'b4': (256, 317, 47, 99),
        'a4': (256, 317, 0, 46),

        'h3': (180, 249, 410, 480),
        'g3': (180, 249, 342, 409),
        'f3': (183, 249, 281, 341),
        'e3': (185, 249, 220, 280),
        'd3': (187, 252, 158, 219),
        'c3': (190, 254, 95, 157),
        'b3': (192, 255, 45, 94),
        'a3': (192, 255, 0, 44),

        'h2': (113, 179, 413, 480),
        'g2': (112, 179, 341, 412),
        'f2': (109, 182, 280, 340),
        'e2': (112, 184, 220, 279),
        'd2': (116, 186, 160, 219),
        'c2': (120, 189, 100, 159),
        'b2': (128, 191, 45, 99),
        'a2': (130, 191, 0, 44),

        'h1': (20, 112, 410, 480),
        'g1': (23, 111, 344, 409),
        'f1': (25, 108, 280, 343),
        'e1': (32, 111, 220, 279),
        'd1': (37, 115, 156, 219),
        'c1': (42, 119, 95, 155),
        'b1': (55, 127, 43, 94),
        'a1': (60, 130, 0, 43),
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
                        if key not in pieces_positions:
                            pieces_positions[key] = []
                        pieces_positions[key].append((x, y, w, h, color))

    return pieces_positions

# If a piece habitates more than one square, the function will determine the most likely position
def determine_piece_position(pieces_positions):
    final_positions = {}
    for key, positions in pieces_positions.items():
        if len(positions) == 1:
            final_positions[key] = positions[0]
        else:
            max_area = 0
            best_position = None
            for (x, y, w, h, color) in positions:
                area = w * h
                if area > max_area:
                    max_area = area
                    best_position = (x, y, w, h, color)
            final_positions[key] = best_position
    return final_positions

# Function to compare position to determine if movement has occurred
def compare_piece_positions(old_positions, new_positions):
    movements = []

    # Track which pieces have moved
    for old_key, old_piece in old_positions.items():
        found = False
        for new_key, new_piece in new_positions.items():
            # Check if the piece color matches
            if old_piece[4] == new_piece[4]:
                found = True
                if old_key != new_key:  # Positions are different, so the piece has moved
                    movements.append((old_key, new_key, old_piece[4]))
                break
        if not found:  # The piece has moved from old_key
            movements.append((old_key, None, old_piece[4]))

    # Check for new pieces that were not in old positions
    for new_key, new_piece in new_positions.items():
        found = False
        for old_key, old_piece in old_positions.items():
            if new_piece[4] == old_piece[4]:
                found = True
                break
        if not found:  # The piece has moved to new_key
            movements.append((None, new_key, new_piece[4]))

    return movements

def generate_movement_strings(movements, old_positions, new_positions):
    move_strings = []
    for move in movements:
        from_pos, to_pos, color = move
        if from_pos and to_pos:
            move_strings.append(f'{from_pos}{to_pos}')
            if to_pos in old_positions and old_positions[to_pos][4] != color:
                move_strings.append(f'{old_positions[to_pos][4][0].upper()} captured at {to_pos}')
        elif from_pos:
            # Piece moved from from_pos but not tracked to a new position
            for new_key, new_piece in new_positions.items():
                if new_piece[4] == color and new_key != from_pos:
                    move_strings.append(f'{from_pos}{new_key}')
                    if new_key in old_positions and old_positions[new_key][4] != color:
                        move_strings.append(f'{old_positions[new_key][4][0].upper()} captured at {new_key}')
                    break
        elif to_pos:
            # New piece appeared at to_pos
            for old_key, old_piece in old_positions.items():
                if old_piece[4] == color and old_key != to_pos:
                    move_strings.append(f'{old_key}{to_pos}')
                    if to_pos in old_positions and old_positions[to_pos][4] != color:
                        move_strings.append(f'{old_positions[to_pos][4][0].upper()} captured at {to_pos}')
                    break

    return move_strings

previous_positions = None

# Open the camera
camera = cv2.VideoCapture(0)
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
cv2.setTrackbarPos('S min', 'HSV Orange Mask', 191)
cv2.setTrackbarPos('S max', 'HSV Orange Mask', 255)
cv2.setTrackbarPos('V min', 'HSV Orange Mask', 105)
cv2.setTrackbarPos('V max', 'HSV Orange Mask', 255)

cv2.setTrackbarPos('H min', 'HSV Black Mask', 89)     
cv2.setTrackbarPos('H max', 'HSV Black Mask', 123)
cv2.setTrackbarPos('S min', 'HSV Black Mask', 0)
cv2.setTrackbarPos('S max', 'HSV Black Mask', 206)
cv2.setTrackbarPos('V min', 'HSV Black Mask', 0)
cv2.setTrackbarPos('V max', 'HSV Black Mask', 82)


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