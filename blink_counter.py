import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indices
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))


def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR)."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# Start video capture
cap = cv2.VideoCapture(0)
blink_count = 0
blink_threshold = 0.2
frame_check = 3
closed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE]
        )
        right_eye = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE]
        )

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Draw eyes
        cv2.polylines(frame, [left_eye], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [right_eye], isClosed=True, color=(0, 255, 0), thickness=2)

        # Blink detection logic
        if avg_ear < blink_threshold:
            closed_frames += 1
        else:
            if closed_frames >= frame_check:
                blink_count += 1
            closed_frames = 0

        # Display EAR and blink count
        cv2.putText(
            frame,
            f"EAR: {avg_ear:.2f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    # Show frame
    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
