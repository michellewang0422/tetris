import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
import pyautogui

# Path to the gesture recognition model
model_path = "gesture_recognizer.task"  # Update this to the correct path where the model is saved

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

# Initialize Mediapipe hands for custom pointing gesture detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to detect if the index finger is definitively pointing right or left
def recognize_pointing(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]  # Index finger base
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Check if the index finger is definitively angled to the side
    index_vector = (index_tip.x - index_mcp.x, index_tip.y - index_mcp.y)
    finger_slope = index_vector[1] / index_vector[0] if index_vector[0] != 0 else float('inf')  # Slope of the finger

    if abs(finger_slope) < 0.3:  # Ensure the finger is roughly horizontal (adjust threshold as needed)
        if index_tip.x > wrist.x:
            return "Pointing Right"
        elif index_tip.x < wrist.x:
            return "Pointing Left"
    return None


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally and convert the BGR image to RGB.
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image to a Mediapipe Image object for the gesture recognizer
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Perform gesture recognition on the image
            result = gesture_recognizer.recognize(mp_image)

            # Convert the image to hand landmarks to detect pointing gestures
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if result.gestures:
                printed_gesture = None
                recognized_gesture = result.gestures[0][0].category_name
                confidence = result.gestures[0][0].score

                # Recognize pointing gestures with hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        pointing_direction = recognize_pointing(hand_landmarks)
                        if pointing_direction == "Pointing Right":
                            pyautogui.press("right")
                        elif pointing_direction == "Pointing Left":
                            pyautogui.press("left")
                        
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        printed_gesture = pointing_direction

                # Example of pressing keys with pyautogui based on other recognized gestures
                if recognized_gesture == "Open_Palm":
                    printed_gesture = recognized_gesture
                    pyautogui.press("up")
                elif recognized_gesture == "Thumb_Down":
                    printed_gesture = recognized_gesture
                    pyautogui.press("down")
                elif recognized_gesture == "Victory":
                    printed_gesture = recognized_gesture
                    pyautogui.press("space")

                pyautogui.PAUSE=0.3

                # Display recognized gesture and confidence 
                cv2.putText(image, f"Gesture: {printed_gesture} ({confidence:.2f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


            # Display the resulting image (can comment this out for better performance later on)
            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to quit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
