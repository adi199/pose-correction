import cv2
import mediapipe.python.solutions.pose as mp_pose
import PoseUtils
import math


class Lunges:
    def __init__(self):
        self.activity_name = 'Bicep Curl'
        self.wrong_pose = {
            'LEFT_KNEE': False,
            'RIGHT_KNEE': False,
        }
        self.correction_message = {
            'knee': {
                'left': '',
                'right': ''
            }
        }
        self.image = None
        self.landmarks = None
        self.target_landmarks = ['LEFT_KNEE', 'RIGHT_KNEE']
        self.rep = 0
        self.movement_direction = 0

    def check_pose(self, image, pose):
        try:
            self.image = image
            self.landmarks = pose.pose_landmarks.landmark
            left_angles = self.get_left_angles()
            right_angles = self.get_right_angles()
            self.evaluate(left_angles[0], right_angles[0])
            self.mark_target_landmarks()
            self.display_angles(left_angles, right_angles)
            self.display_reps()
        except Exception as e:
            print(e)

    def display_reps(self):
        cv2.putText(self.image, str(math.ceil(self.rep)), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (127, 255, 127), 2)

    def display_angles(self, left_angles, right_angles):
        text_face = cv2.FONT_HERSHEY_DUPLEX
        text_scale, text_thickness = 0.5, 2

        text = str(math.ceil(left_angles[0]))
        text_size = cv2.getTextSize(text, text_face, text_scale, text_thickness)
        center = [*self.get_coord('LEFT_KNEE'), ]
        center[0] = int(center[0] * self.image.shape[1])
        center[1] = int(center[1] * self.image.shape[0])
        center = (center[0] - text_size[0][0] // 2, center[1] + text_size[0][1] // 2)
        cv2.putText(self.image, text, center, text_face, text_scale, (127, 255, 127), 1, cv2.LINE_AA)

        text = str(math.ceil(right_angles[0]))
        text_size = cv2.getTextSize(text, text_face, text_scale, text_thickness)
        center = [*self.get_coord('RIGHT_KNEE'), ]
        center[0] = int(center[0] * self.image.shape[1])
        center[1] = int(center[1] * self.image.shape[0])
        center = (center[0] - text_size[0][0] // 2, center[1] + text_size[0][1] // 2)
        cv2.putText(self.image, text, center, text_face, text_scale, (127, 255, 127), 1, cv2.LINE_AA)

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
        left_knee = self.get_coord('LEFT_KNEE')
        left_ankle = self.get_coord('LEFT_ANKLE')

        left_knee_angle = PoseUtils.calculate_angle(left_hip, left_knee, left_ankle)

        return [left_knee_angle]

    def get_right_angles(self):
        right_hip = self.get_coord('RIGHT_HIP')
        right_knee = self.get_coord('RIGHT_KNEE')
        right_ankle = self.get_coord('RIGHT_ANKLE')

        right_shoulder_angle = PoseUtils.calculate_angle(right_hip, right_knee, right_ankle)

        return [right_shoulder_angle]

    def evaluate(self, left_knee_angle, right_knee_angle):
        if (left_knee_angle < 90 or right_knee_angle < 90) and self.movement_direction == 0:
            self.rep += 0.5
            self.movement_direction = 1
        if (left_knee_angle > 170 or right_knee_angle > 170) and self.movement_direction == 1:
            self.rep += 0.5
            self.movement_direction = 0

        if 80.0 > left_knee_angle:
            self.correction_message['knee']['left'] = 'Excess Left knee bent'
            self.wrong_pose['LEFT_KNEE'] = True
        else:
            self.correction_message['knee']['left'] = ''
            self.wrong_pose['LEFT_KNEE'] = False

        if 80.0 > right_knee_angle:
            self.correction_message['knee']['right'] = 'Excess Right knee bent'
            self.wrong_pose['RIGHT_KNEE'] = True
        else:
            self.correction_message['knee']['right'] = ''
            self.wrong_pose['RIGHT_KNEE'] = False

    def get_coord(self, landmark_name):
        landmark = self.landmarks[mp_pose.PoseLandmark[landmark_name]]
        return [landmark.x, landmark.y]

    def get_correction_message(self):
        return self.correction_message
