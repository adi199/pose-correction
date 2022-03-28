import cv2 as cv
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_draw
import PoseUtils
from BicepCurl import BicepCurl


class PoseEstimation:

    feed_name = 'Webcam Feed'
    frame = None
    pose = None

    def __init__(self, min_detection_confidence, min_tracking_confidence, input_frame=None):
        # Connection to webcam
        if input_frame is None:
            self.web_cam = cv.VideoCapture(0)
        else:
            self.frame = cv.imread(input_frame)

        self.pose_obj = mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                     min_tracking_confidence=min_tracking_confidence)

    def estimate(self):
        self.estimate_pose()
        self.draw_pose()

        try:
            bicep_curl = BicepCurl()
            bicep_curl.check_pose(self.pose)
            message = bicep_curl.get_correction_message()
            self.put_text((10, 20), message['shoulder']['left'])
            self.put_text((10, 40), message['shoulder']['right'])
        except Exception as e:
            print(e)
            pass

        cv.imshow(winname='Image', mat=self.frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def start_feed(self):
        bicep_curl = BicepCurl()
        while self.web_cam.isOpened():
            # Read the frame
            ret, self.frame = self.web_cam.read()

            # Get and draw pose in the frame
            self.estimate_pose()
            self.draw_pose()

            try:
                bicep_curl.check_pose(self.pose)
                message = bicep_curl.get_correction_message()
                self.put_text((10, 20), message['shoulder']['left'])
                self.put_text((10, 40), message['shoulder']['right'])
            except Exception as e:
                print(e)
                pass

            # Exit condition
            if cv.waitKey(10) & 0xFF == ord('q'):
                self.close_feed()
                break

            # Display frame
            cv.imshow(winname=self.feed_name, mat=self.frame)

    def put_text(self, coord, text):
        cv.putText(self.frame, str(text), coord, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

    def draw_pose(self):
        mp_draw.draw_landmarks(self.frame, self.pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    def close_feed(self):
        self.web_cam.release()
        cv.destroyAllWindows()

    def estimate_pose(self):
        rgb_frame = PoseUtils.get_rgb_from_bgr(self.frame)
        self.pose = self.pose_obj.process(rgb_frame)


if __name__ == '__main__':
    pose_estimation = PoseEstimation(min_detection_confidence=0.5, min_tracking_confidence=0.5, input_frame='./data/sample.jpeg')
    pose_estimation.estimate()
