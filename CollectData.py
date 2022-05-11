import cv2 as cv
import PoseUtils
import mediapipe.python.solutions.pose as mp_pose
import pandas as pd
import traceback


def get_coord(landmarks, landmark_name):
    landmark = landmarks[mp_pose.PoseLandmark[landmark_name]]
    return [landmark.x, landmark.y]


def get_label(left, right):
    label = ''
    if 35.0 <= left:
        label += '0'

    if 35.0 <= right:
        label += '1'

    return 0 if label == '0' else 1 if label == '1' else 2 if label == '01' else 3


if __name__ == '__main__':
    data = {
        'LEFT_SHOULDER': [],
        'RIGHT_SHOULDER': [],
        'LABEL': []
    }
    pose_obj = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    web_cam = cv.VideoCapture(0)
    time = 0
    while web_cam.isOpened():
        try:
            ret, frame = web_cam.read()
            rgb_frame = PoseUtils.get_rgb_from_bgr(frame)
            pose = pose_obj.process(rgb_frame)
            landmarks = pose.pose_landmarks.landmark

            left_hip = get_coord(landmarks, 'LEFT_HIP')
            left_shoulder = get_coord(landmarks, 'LEFT_SHOULDER')
            left_elbow = get_coord(landmarks, 'LEFT_ELBOW')

            left_shoulder_angle = PoseUtils.calculate_angle(left_hip, left_shoulder, left_elbow)

            right_hip = get_coord(landmarks, 'RIGHT_HIP')
            right_shoulder = get_coord(landmarks, 'RIGHT_SHOULDER')
            right_elbow = get_coord(landmarks, 'RIGHT_ELBOW')

            right_shoulder_angle = PoseUtils.calculate_angle(right_hip, right_shoulder, right_elbow)

            data['LEFT_SHOULDER'].append(left_shoulder_angle)
            data['RIGHT_SHOULDER'].append(right_shoulder_angle)
            data['LABEL'].append(get_label(left_shoulder_angle, right_shoulder_angle))

            cv.imshow(winname='web_cam', mat=frame)

            if cv.waitKey(10) & 0xFF == ord('q'):
                web_cam.release()
                cv.destroyAllWindows()
                break

            if time == 1000:
                break

            time += 1
        except Exception as e:
            print(traceback.format_exc())
    data = pd.DataFrame(data)
    data.to_csv('input_data.csv', index=False)


