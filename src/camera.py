import cv2


def launch_camera():
    stream = cv2.VideoCapture(0)

    if not stream.isOpened():
        print("Error: Could not open video stream")
        return

    try:
        while True:
            ret, frame = stream.read()
            if not ret:
                print("Error: Could not read frame")
                break

            cv2.imshow("Video Stream", frame)

            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    launch_camera()
