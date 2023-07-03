# We could use the std math library and it is faster in our case but for the sake of SIMD we will be using numpy (Optimization for later uses)
# Using a class for later imports
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapipe as mp
import math
import globals

# We could use the std math library and it is faster in our case but for the sake of SIMD we will be using numpy (Optimization for later uses)
# Using a class for later imports


class Pose_Detection_Toolkit:

    def distance(self, vector1, vector2):
        return np.linalg.norm(vector1 - vector2)  # Euclidean Norm / Norme 2

    def norm(self, vector):
        assert isinstance(vector, np.ndarray), 'input must be np array'
        return np.linalg.norm(vector)

    # Works in both 2D and 3D
    def angle_between_vectors(self, vector1, vector2):
        # A . B = norm(A) * norm(B) * cos(angle_between_vectors(A, B))
        # => arccos(angle_between_vectors(A, B)) = A.B / (norm(A) * norm(B))
        norm_A = np.linalg.norm(vector1)
        norm_B = np.linalg.norm(vector2)
        assert norm_A * norm_B != 0, 'Norms should be != 0'
        # Angle in deg not in rad
        return np.arccos(np.dot(vector1, vector2) / (norm_A * norm_B)) * 180 / np.pi
        # arcos : [-1,1] --> [0,pi]

    def generate_vectors(self, results, image_height, image_width):
        # Needed vectors : Both left and right
        # Shoulder-Elbow
        # Elbow-Wrist
        # Wrist-Index
        # Shoulder-Hip
        # Hip-Knee
        # Knee-Ankle
        # Ankle-FootIndex

        landmarks = results.pose_landmarks
        mph_landmarks = mp_holistic.PoseLandmark

        shoulder_elbow_left_x = landmarks.landmark[mph_landmarks.LEFT_ELBOW].x - \
            landmarks.landmark[mph_landmarks.LEFT_SHOULDER].x
        shoulder_elbow_right_x = landmarks.landmark[mph_landmarks.RIGHT_ELBOW].x - \
            landmarks.landmark[mph_landmarks.RIGHT_SHOULDER].x
        shoulder_elbow_left_y = landmarks.landmark[mph_landmarks.LEFT_ELBOW].y - \
            landmarks.landmark[mph_landmarks.LEFT_SHOULDER].y
        shoulder_elbow_right_y = landmarks.landmark[mph_landmarks.RIGHT_ELBOW].y - \
            landmarks.landmark[mph_landmarks.RIGHT_SHOULDER].y

        elbow_wrist_left_y = landmarks.landmark[mph_landmarks.LEFT_WRIST].y - \
            landmarks.landmark[mph_landmarks.LEFT_ELBOW].y
        elbow_wrist_right_y = landmarks.landmark[mph_landmarks.RIGHT_WRIST].y - \
            landmarks.landmark[mph_landmarks.RIGHT_ELBOW].y
        elbow_wrist_left_x = landmarks.landmark[mph_landmarks.LEFT_WRIST].x - \
            landmarks.landmark[mph_landmarks.LEFT_ELBOW].x
        elbow_wrist_right_x = landmarks.landmark[mph_landmarks.RIGHT_WRIST].x - \
            landmarks.landmark[mph_landmarks.RIGHT_ELBOW].x

        wrist_index_left_x = landmarks.landmark[mph_landmarks.LEFT_WRIST].x - \
            landmarks.landmark[mph_landmarks.LEFT_INDEX].x
        wrist_index_right_x = landmarks.landmark[mph_landmarks.RIGHT_WRIST].x - \
            landmarks.landmark[mph_landmarks.RIGHT_INDEX].x
        wrist_index_left_y = landmarks.landmark[mph_landmarks.LEFT_WRIST].y - \
            landmarks.landmark[mph_landmarks.LEFT_INDEX].y
        wrist_index_right_y = landmarks.landmark[mph_landmarks.RIGHT_WRIST].y - \
            landmarks.landmark[mph_landmarks.RIGHT_INDEX].y

        shoulder_hip_left_x = landmarks.landmark[mph_landmarks.LEFT_SHOULDER].x - \
            landmarks.landmark[mph_landmarks.LEFT_HIP].x
        shoulder_hip_right_x = landmarks.landmark[mph_landmarks.RIGHT_SHOULDER].x - \
            landmarks.landmark[mph_landmarks.RIGHT_HIP].x
        shoulder_hip_left_y = landmarks.landmark[mph_landmarks.LEFT_SHOULDER].y - \
            landmarks.landmark[mph_landmarks.LEFT_HIP].y
        shoulder_hip_right_y = landmarks.landmark[mph_landmarks.RIGHT_SHOULDER].y - \
            landmarks.landmark[mph_landmarks.RIGHT_HIP].y

        hip_knee_left_x = landmarks.landmark[mph_landmarks.LEFT_HIP].x - \
            landmarks.landmark[mph_landmarks.LEFT_KNEE].x
        hip_knee_left_y = landmarks.landmark[mph_landmarks.LEFT_HIP].y - \
            landmarks.landmark[mph_landmarks.LEFT_KNEE].y
        hip_knee_right_x = landmarks.landmark[mph_landmarks.RIGHT_HIP].x - \
            landmarks.landmark[mph_landmarks.RIGHT_KNEE].x
        hip_knee_right_y = landmarks.landmark[mph_landmarks.RIGHT_HIP].y - \
            landmarks.landmark[mph_landmarks.RIGHT_KNEE].y

        knee_ankle_left_x = landmarks.landmark[mph_landmarks.LEFT_ANKLE].x - \
            landmarks.landmark[mph_landmarks.LEFT_KNEE].x
        knee_ankle_left_y = landmarks.landmark[mph_landmarks.LEFT_ANKLE].y - \
            landmarks.landmark[mph_landmarks.LEFT_KNEE].y
        knee_ankle_right_x = landmarks.landmark[mph_landmarks.RIGHT_ANKLE].x - \
            landmarks.landmark[mph_landmarks.RIGHT_KNEE].x
        knee_ankle_right_y = landmarks.landmark[mph_landmarks.RIGHT_ANKLE].y - \
            landmarks.landmark[mph_landmarks.RIGHT_KNEE].y

        footindex_ankle_left_x = landmarks.landmark[mph_landmarks.LEFT_ANKLE].x - \
            landmarks.landmark[mph_landmarks.LEFT_FOOT_INDEX].x
        footindex_ankle_left_y = landmarks.landmark[mph_landmarks.LEFT_ANKLE].y - \
            landmarks.landmark[mph_landmarks.LEFT_FOOT_INDEX].y
        footindex_ankle_right_x = landmarks.landmark[mph_landmarks.RIGHT_ANKLE].x - \
            landmarks.landmark[mph_landmarks.RIGHT_FOOT_INDEX].x
        footindex_ankle_right_y = landmarks.landmark[mph_landmarks.RIGHT_ANKLE].y - \
            landmarks.landmark[mph_landmarks.RIGHT_FOOT_INDEX].y

        right_shoulder_left_shoulder_x = landmarks.landmark[mph_landmarks.LEFT_SHOULDER].x - \
            landmarks.landmark[mph_landmarks.RIGHT_SHOULDER].x
        right_shoulder_left_shoulder_y = landmarks.landmark[mph_landmarks.LEFT_SHOULDER].y - \
            landmarks.landmark[mph_landmarks.RIGHT_SHOULDER].y
        der_left_shoulder_y = results.pose_landmarks.landmark[mph_landmarks.LEFT_SHOULDER].y - \
            results.pose_landmarks.landmark[mph_landmarks.RIGHT_SHOULDER].y

        footindex_heel_left_x = landmarks.landmark[mph_landmarks.LEFT_FOOT_INDEX].x - \
            landmarks.landmark[mph_landmarks.LEFT_HEEL].x
        footindex_heel_left_y = landmarks.landmark[mph_landmarks.LEFT_FOOT_INDEX].y - \
            landmarks.landmark[mph_landmarks.LEFT_HEEL].y

        footindex_heel_right_x = landmarks.landmark[mph_landmarks.RIGHT_FOOT_INDEX].x - \
            landmarks.landmark[mph_landmarks.RIGHT_HEEL].x
        footindex_heel_right_y = landmarks.landmark[mph_landmarks.RIGHT_FOOT_INDEX].y - \
            landmarks.landmark[mph_landmarks.RIGHT_HEEL].y

        return {
            "shoulder_elbow": np.array([[shoulder_elbow_left_x * image_width, shoulder_elbow_left_y * image_height],
                                        [shoulder_elbow_right_x * image_width, shoulder_elbow_right_y * image_height]]),
            "elbow_wrist": np.array([[elbow_wrist_left_x * image_width, elbow_wrist_left_y * image_height],
                                     [elbow_wrist_right_x * image_width, elbow_wrist_right_y * image_height]]),
            "wrist_index": np.array([[wrist_index_left_x * image_width, wrist_index_left_y * image_height],
                                     [wrist_index_right_x * image_width, wrist_index_right_y * image_height]]),
            "shoulder_hip": np.array([[shoulder_hip_left_x * image_width, shoulder_hip_left_y * image_height],
                                      [shoulder_hip_right_x * image_width, shoulder_hip_right_y * image_height]]),
            "hip_knee": np.array([[hip_knee_left_x * image_width, hip_knee_left_y * image_height],
                                  [hip_knee_right_x * image_width, hip_knee_right_y * image_height]]),
            "knee_ankle": np.array([[knee_ankle_left_x * image_width, knee_ankle_left_y * image_height],
                                    [knee_ankle_right_x * image_width, knee_ankle_right_y * image_height]]),
            "footindex_ankle": np.array([[footindex_ankle_left_x * image_width, footindex_ankle_left_y * image_height],
                                         [footindex_ankle_right_x * image_width, footindex_ankle_right_y * image_height]]),
            "right_shoulder_left_shoulder": np.array([right_shoulder_left_shoulder_x * image_width, right_shoulder_left_shoulder_y * image_height]),
            "footindex_heel": np.array([[footindex_heel_left_x * image_width, footindex_heel_left_y * image_height], [footindex_heel_right_x * image_width, footindex_heel_right_y * image_height]])
        }
        # [0] for the left
        # [1] for the right

    def is_backward(self, vectors):  # Detect whether an image is backward or forward
        return vectors['right_shoulder_left_shoulder'][0] < 0

    # Need to find a way to get a more accurate angle range
    def Pose_Dectection(self, vectors):
        # The needed angles are the following :
        # Between the shoulder_elbow vector and the shoulder_hip vector
        angle_se_sh_left = self.angle_between_vectors(
            vectors['shoulder_elbow'][0], -vectors['shoulder_hip'][0])
        angle_se_sh_right = self.angle_between_vectors(
            vectors['shoulder_elbow'][1], -vectors['shoulder_hip'][1])
        # Between the hip_knee vector and the shoulder_hip vector
        angle_hk_sh_left = self.angle_between_vectors(
            vectors['hip_knee'][0], -vectors['shoulder_hip'][0])
        angle_hk_sh_right = self.angle_between_vectors(
            vectors['hip_knee'][1], -vectors['shoulder_hip'][1])
        # Between the elbow_wrist vector and the shoulder_elbow vector
        angle_ew_se_left = self.angle_between_vectors(
            vectors['elbow_wrist'][0], -vectors['shoulder_elbow'][0])
        angle_ew_se_right = self.angle_between_vectors(
            vectors['elbow_wrist'][1], -vectors['shoulder_elbow'][1])
        # Between the hip_knee vector and the knee_ankle vector
        angle_hk_ka_left = self.angle_between_vectors(
            vectors['hip_knee'][0], vectors['knee_ankle'][0])
        angle_hk_ka_right = self.angle_between_vectors(
            vectors['hip_knee'][1], vectors['knee_ankle'][1])

        # Verifying the shoulders and the knees angles
        if 150 < angle_ew_se_left < 180 and 150 < angle_ew_se_right < 180 and 140 < angle_hk_ka_left < 180 and 140 < angle_hk_ka_right < 180:
            if 40 < angle_se_sh_left < 70 and 40 < angle_se_sh_right < 70:
                return "A Pose"
            elif 70 <= angle_se_sh_left < 110 and 70 <= angle_se_sh_right < 110:
                return "T Pose"
        return "Unknown Pose"

    # Checks whether the body in our image is turning to the left or the right
    def image_orientation(self, results, vectors):
        landmarks = results.pose_landmarks
        mph_landmarks = mp_holistic.PoseLandmark

        right_shoulder_left_shoulder_x = landmarks.landmark[mph_landmarks.RIGHT_SHOULDER].x - \
            landmarks.landmark[mph_landmarks.LEFT_SHOULDER].x
        right_shoulder_left_shoulder_y = landmarks.landmark[mph_landmarks.RIGHT_SHOULDER].y - \
            landmarks.landmark[mph_landmarks.LEFT_SHOULDER].y
        right_shoulder_left_shoulder_z = landmarks.landmark[mph_landmarks.RIGHT_SHOULDER].z - \
            landmarks.landmark[mph_landmarks.LEFT_SHOULDER].z
        right_shoulder_left_shoulder = np.array(
            [right_shoulder_left_shoulder_x, right_shoulder_left_shoulder_y, right_shoulder_left_shoulder_z])

        angle = self.angle_between_vectors(
            right_shoulder_left_shoulder, np.array([1, 0, 0]))

        # This value is set by observation of some examples of correct and icorrect poses and can/should be corrected
        hand_threshold = 0.10898104310035706

        hand_test = 0.105 <= max(abs(landmarks.landmark[mph_landmarks.RIGHT_THUMB].x - landmarks.landmark[mph_landmarks.RIGHT_HIP].x),
                                 abs(landmarks.landmark[mph_landmarks.LEFT_THUMB].x - landmarks.landmark[mph_landmarks.LEFT_HIP].x)) <= 0.109

        ans = []

        if 85 < angle < 95:
            if landmarks.landmark[mph_landmarks.NOSE].x > landmarks.landmark[mph_landmarks.RIGHT_SHOULDER].x and landmarks.landmark[mph_landmarks.NOSE].x > landmarks.landmark[mph_landmarks.LEFT_SHOULDER].x:
                ans = ["RIGHT", hand_test]
            else:
                ans = ["LEFT", hand_test]
        else:
            ans = ["NO ORIENTATION", hand_test]

        # The test is to check whether the hands position is good or not
        return ans

    def is_valid(self, results, vectors):
        landmarks = results.pose_landmarks

        if landmarks is None:
            return False
        required_landmarks = [
            mp_holistic.PoseLandmark.RIGHT_EYE,
            mp_holistic.PoseLandmark.LEFT_EYE,
            mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX,
            mp_holistic.PoseLandmark.LEFT_FOOT_INDEX
        ]

        for landmark in required_landmarks:
            x = landmarks.landmark[landmark].x
            y = landmarks.landmark[landmark].y

            if x < 0 or x > 1 or y < 0 or y > 1:  # Whether it is outside of the image or collapsed
                return False

        [orientation, hand_test] = self.image_orientation(results, vectors)
        if orientation == 'LEFT' or orientation == 'RIGHT':
            if hand_test == False:
                return False

        return True

    

class Body:
    def __init__(self, **kwargs):
        self._left_eye = kwargs.get('left_eye', None)
        self._right_eye = kwargs.get('right_eye', None)
        self._nose = kwargs.get('nose', None)
        self._left_shoulder = kwargs.get('left_shoulder', None)
        self._right_shoulder = kwargs.get('right_shoulder', None)
        self._left_elbow = kwargs.get('left_elbow', None)
        self._right_elbow = kwargs.get('right_elbow', None)
        self._left_wrist = kwargs.get('left_wrist', None)
        self._right_wrist = kwargs.get('right_wrist', None)
        self._left_hip = kwargs.get('left_hip', None)
        self._right_hip = kwargs.get('right_hip', None)
        self._left_knee = kwargs.get('left_knee', None)
        self._right_knee = kwargs.get('right_knee', None)
        self._left_ankle = kwargs.get('left_ankle', None)
        self._right_ankle = kwargs.get('right_ankle', None)
        self._left_heel = kwargs.get('left_heel', None)
        self._right_heel = kwargs.get('right_heel', None)
        self._left_foot_index = kwargs.get('left_foot_index', None)
        self._right_foot_index = kwargs.get('right_foot_index', None)

    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)

    def get_attributes(self):
        return {
            'left_eye': self._left_eye,
            'right_eye': self._right_eye,
            'nose': self._nose,
            'left_shoulder': self._left_shoulder,
            'right_shoulder': self._right_shoulder,
            'left_elbow': self._left_elbow,
            'right_elbow': self._right_elbow,
            'left_wrist': self._left_wrist,
            'right_wrist': self._right_wrist,
            'left_hip': self._left_hip,
            'right_hip': self._right_hip,
            'left_knee': self._left_knee,
            'right_knee': self._right_knee,
            'left_ankle': self._left_ankle,
            'right_ankle': self._right_ankle,
            'left_heel': self._left_heel,
            'right_heel': self._right_heel,
            'left_foot_index': self._left_foot_index,
            'right_foot_index': self._right_foot_index
        }


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose


def process(image=None):

    IMAGE_FILES = []

    # For static images:
    if image is not None:
        IMAGE_FILES.append(image)

    else:
        return None

    pdtk = Pose_Detection_Toolkit()

    BG_COLOR = (192, 192, 192)  # gray
    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True) as holistic:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks is None:
                print(f"No pose landmarks detected in {file}. Skipping...")
                continue

            # for attr, value in attributes.items():
            #     print(f"{attr}: {value}")

            vectors = pdtk.generate_vectors(results, image_height, image_width)

            # print(f'is my image backward ? : {pdtk.is_backward(vectors)}')
            # print(f'is my image valid ? does it contain all the needed points ? : {pdtk.is_valid(results)}')
            # print(f'In what pose is my image ? : {pdtk.Pos_Dectection(vectors)}')

            condition = 0
            annotated_image = image.copy()
            if results.segmentation_mask is not None:
                condition = np.stack(
                    (results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose, left and right hands, and face landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.
                get_default_pose_landmarks_style())
            plt.imshow(image)

            body = Body()

            if results.pose_landmarks is not None:
                body.set_attributes(
                    left_eye=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE],
                    right_eye=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE],
                    nose=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE],
                    left_shoulder=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER],
                    right_shoulder=results.pose_landmarks.landmark[
                        mp_holistic.PoseLandmark.RIGHT_SHOULDER],
                    left_elbow=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW],
                    right_elbow=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW],
                    left_wrist=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST],
                    right_wrist=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST],
                    left_hip=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP],
                    right_hip=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP],
                    left_knee=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE],
                    right_knee=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE],
                    left_ankle=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE],
                    right_ankle=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE],
                    left_heel=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HEEL],
                    right_heel=results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HEEL],
                    left_foot_index=results.pose_landmarks.landmark[
                        mp_holistic.PoseLandmark.LEFT_FOOT_INDEX],
                    right_foot_index=results.pose_landmarks.landmark[
                        mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX]
                )

            attributes = body.get_attributes()
            infos = {
                'body landmarks': attributes,
                'is_backward': pdtk.is_backward(vectors),
                'orientation': pdtk.image_orientation(results, vectors),
                'is_valid': pdtk.is_valid(results, vectors),
                'pose': pdtk.Pose_Dectection(vectors),
                'vectors': vectors,
                'image_height': image.shape[0],
                'image_width': image.shape[1],
            }

            return body, infos
