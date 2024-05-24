import cv2

# Open the camera
camera = cv2.VideoCapture(0) # Camera index, might be different on other devices
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Bring up the camera settings window
camera.set(cv2.CAP_PROP_SETTINGS, 1)

cv2.namedWindow('Camera Feed')

while True:
    ret, frame = camera.read()
    if not ret:
        break

    cv2.imshow('Camera Feed', frame)

    # Check for key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close the windows
camera.release()
cv2.destroyAllWindows()
