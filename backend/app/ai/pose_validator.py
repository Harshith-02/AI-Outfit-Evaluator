import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5
)

def validate_full_body(image_path):

    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return {
            "full_body_detected": False,
            "message": "No pose detected"
        }

    landmarks = results.pose_landmarks.landmark

    required_points = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]

    missing_points = []

    for point in required_points:

        visibility = landmarks[point].visibility

        if visibility < 0.5:
            missing_points.append(point.name)

    return {
        "full_body_detected": len(missing_points) == 0,
        "missing_points": missing_points
    }