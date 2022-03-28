import mediapipe.python.solutions.pose as mp_pose
import PoseUtils


class BicepCurl:
    activity_name = 'Bicep Curl'
    correction_message = {
        'shoulder': {
            'left': '',
            'right': ''
        }
    }
    landmarks = None

    def check_pose(self, pose):
        try:
            self.landmarks = pose.pose_landmarks.landmark
            left_angles = self.get_left_angles()
            right_angles = self.get_right_angles()
            self.evaluate(left_angles[0], left_angles[1], right_angles[0], right_angles[1])
        except Exception as e:
            print(e)

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

    def evaluate(self, left_shoulder_angle, left_elbow_angle, right_shoulder_angle, right_elbow_angle):
        if 35.0 <= left_shoulder_angle:
            self.correction_message['shoulder']['left'] = 'Excess Left arm rotation'
        else:
            self.correction_message['shoulder']['left'] = ''

        if 35.0 <= right_shoulder_angle <= 70.0:
            self.correction_message['shoulder']['right'] = 'Excess Right arm rotation'
        else:
            self.correction_message['shoulder']['right'] = ''

    def get_coord(self, landmark_name):
        return [self.landmarks[mp_pose.PoseLandmark[landmark_name]].x,
                self.landmarks[mp_pose.PoseLandmark[landmark_name]].y]

    def get_correction_message(self):
        return self.correction_message
