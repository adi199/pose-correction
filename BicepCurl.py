import cv2
import mediapipe.python.solutions.pose as mp_pose
import PoseUtils
import math
import traceback
from Model import Model


class BicepCurl:
    def __init__(self):
        self.activity_name = 'Bicep Curl'
        self.wrong_pose = {
            'LEFT_SHOULDER': False,
            'LEFT_ELBOW': False,
            'RIGHT_SHOULDER': False,
            'RIGHT_ELBOW': False,
        }
        self.correction_message = {
            'shoulder': {
                'left': '',
                'right': ''
            },
            'elbow': {
                'left': '',
                'right': ''
            }
        }
        self.image = None
        self.landmarks = None
        self.target_landmarks = ['LEFT_ELBOW', 'LEFT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_SHOULDER']
        self.rep = 0
        self.movement_direction = 1
        self.model = Model()

    def check_pose(self, image, pose):
        try:
            self.image = image
            self.landmarks = pose.pose_landmarks.landmark
            left_angles = self.get_left_angles()
            right_angles = self.get_right_angles()
            self.ml_evaluate(left_angles[0], left_angles[1], right_angles[0], right_angles[1])
            self.mark_target_landmarks()
            self.display_angles(left_angles, right_angles)
            self.display_reps()
        except Exception as e:
            print(traceback.format_exc())

    def display_reps(self):
        cv2.putText(self.image, str(math.ceil(self.rep)), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (127, 255, 127), 2)

    def display_angle(self, text, landmark):
        text_face = cv2.FONT_HERSHEY_DUPLEX
        text_scale, text_thickness = 0.5, 2

        text_size = cv2.getTextSize(text, text_face, text_scale, text_thickness)
        center = [*self.get_coord(landmark), ]
        center[0] = int(center[0] * self.image.shape[1])
        center[1] = int(center[1] * self.image.shape[0])
        center = (center[0] - text_size[0][0] // 2, center[1] + text_size[0][1] // 2)
        cv2.putText(self.image, text, center, text_face, text_scale, (127, 255, 127), 1, cv2.LINE_AA)

    def display_angles(self, left_angles, right_angles):
        self.display_angle(str(math.ceil(left_angles[0])), 'LEFT_SHOULDER')
        self.display_angle(str(math.ceil(right_angles[1])), 'LEFT_ELBOW')
        self.display_angle(str(math.ceil(right_angles[0])), 'RIGHT_SHOULDER')
        self.display_angle(str(math.ceil(right_angles[1])), 'RIGHT_ELBOW')

    def mark_target_landmarks(self):
        overlay = self.image.copy()
        for target_landmark in self.target_landmarks:
            center = [*self.get_coord(target_landmark), ]
            center[0] = int(center[0] * self.image.shape[1])
            center[1] = int(center[1] * self.image.shape[0])
            overlay = cv2.circle(overlay, center, 30, self.get_marker_color(target_landmark), -1)
        cv2.addWeighted(self.image, 1, overlay, 0.2, 0, dst=self.image)

    def get_marker_color(self, landmark):
        return (0, 255, 27) if not self.wrong_pose[landmark] else (0, 0, 255)

    def get_left_angles(self):
        left_hip = self.get_coord('LEFT_HIP')
        left_shoulder = self.get_coord('LEFT_SHOULDER')
        left_elbow = self.get_coord('LEFT_ELBOW')
        left_wrist = self.get_coord('LEFT_WRIST')

        left_shoulder_angle = PoseUtils.calculate_angle(left_hip, left_shoulder, left_elbow)
        left_elbow_angle = PoseUtils.calculate_angle(left_shoulder, left_elbow, left_wrist)

        return [left_shoulder_angle, left_elbow_angle]

    def get_right_angles(self):
        right_hip = self.get_coord('RIGHT_HIP')
        right_shoulder = self.get_coord('RIGHT_SHOULDER')
        right_elbow = self.get_coord('RIGHT_ELBOW')
        right_wrist = self.get_coord('RIGHT_WRIST')

        right_shoulder_angle = PoseUtils.calculate_angle(right_hip, right_shoulder, right_elbow)
        right_elbow_angle = PoseUtils.calculate_angle(right_shoulder, right_elbow, right_wrist)

        return [right_shoulder_angle, right_elbow_angle]

    def ml_evaluate(self,  left_shoulder_angle, left_elbow_angle, right_shoulder_angle, right_elbow_angle):
        if (left_elbow_angle >= 175 and right_elbow_angle >= 175) and self.movement_direction == 0:
            self.rep += 0.5
            self.movement_direction = 1
        if (left_elbow_angle <= 7 and right_elbow_angle <= 7) and self.movement_direction == 1:
            self.rep += 0.5
            self.movement_direction = 0

        label = self.model.predict([[left_shoulder_angle, right_shoulder_angle]])

        self.correction_message['shoulder']['left'] = 'Excess Left arm rotation' if label == 0 or label == 2 else ''
        self.wrong_pose['LEFT_SHOULDER'] = True if label == 0 or label == 2 else False

        self.correction_message['shoulder']['right'] = 'Excess Right arm rotation' if label == 1 or label == 2 else ''
        self.wrong_pose['RIGHT_SHOULDER'] = True if label == 1 or label == 2 else False

    def evaluate(self, left_shoulder_angle, left_elbow_angle, right_shoulder_angle, right_elbow_angle):
        if (left_elbow_angle >= 175 and right_elbow_angle >= 175) and self.movement_direction == 0:
            self.rep += 0.5
            self.movement_direction = 1
        if (left_elbow_angle <= 7 and right_elbow_angle <= 7) and self.movement_direction == 1:
            self.rep += 0.5
            self.movement_direction = 0

        if 35.0 <= left_shoulder_angle:
            self.correction_message['shoulder']['left'] = 'Excess Left arm rotation'
            self.wrong_pose['LEFT_SHOULDER'] = True
        else:
            self.correction_message['shoulder']['left'] = ''
            self.wrong_pose['LEFT_SHOULDER'] = False

        if 35.0 <= right_shoulder_angle:
            self.correction_message['shoulder']['right'] = 'Excess Right arm rotation'
            self.wrong_pose['RIGHT_SHOULDER'] = True
        else:
            self.correction_message['shoulder']['right'] = ''
            self.wrong_pose['RIGHT_SHOULDER'] = False

    def get_coord(self, landmark_name):
        landmark = self.landmarks[mp_pose.PoseLandmark[landmark_name]]
        return [landmark.x, landmark.y]

    def get_correction_message(self):
        return self.correction_message
