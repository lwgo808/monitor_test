import cv2

stream = cv2.VideoCapture(0)

if not stream.isOpened():
    print("Error: Could not open video stream")
    exit()

while(True):
    ret, frame = stream.read()
    if not ret:
        print("Error: Could not read frame")
        break

    cv2.imshow('Video Stream', frame)

    if cv2.waitKey(1)  == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()
