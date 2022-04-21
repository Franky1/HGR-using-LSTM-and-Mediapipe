import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow import keras
import av

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

modelF= keras.models.load_model('rec_0.h5')

# DATA_PATH = os.path.join('dynamic_dataset') 
actions = np.array(['hello', 'thanks', 'i love you', 'stop', 'please', 'walk', 'argue', 'yes', 'see', 'good'])
# 40 videos
no_sequences = 80
# 30 frames
sequence_length = 30

# for action in actions: 
#     for sequence in range(no_sequences):
#         try: 
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                 
    results = model.process(image)                
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):

    # left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                            ) 
    # right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            ) 

def extract_keypoints(results):
    key1 = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    key2 = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([key1, key2, lh, rh])

colors = [(245,117,16), (117,245,16), (16,117,245),(16,17,245),(16,117,24),(17,25,160),(11,45,116),(170,205,165), (224, 32, 28), (22,142,100)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame



class OpenCamera (VideoProcessorBase):
    def __init__(self) -> None :
        self.sequence = []
        self.sentence = []
        self.threshold = 0.4
    def recv(self, frame):
        img=frame.to_ndarray(format="bgr24")
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = mediapipe_detection(img,holistic)
            draw_styled_landmarks(image, results)
            # 2. Prediction logic
            keypoints = extract_keypoints(results)

            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]
            
            if len(self.sequence) == 30:
                res = modelF.predict(np.expand_dims(self.sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                
                
            #3. Viz logic
                if res[np.argmax(res)] > self.threshold: 
                    if len(self.sentence) > 0: 
                        if actions[np.argmax(res)] != self.sentence[-1]:
                           self.sentence.append(actions[np.argmax(res)])
                    else:
                        self.sentence.append(actions[np.argmax(res)])

                if len(self.sentence) > 1: 
                    self.sentence = self.sentence[-1:]

                    # Viz probabilities
                    # image = prob_viz(res, actions, image, colors)
            
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(self.sentence), (3,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
    #   av.VideoFrame.from_ndarray(image, format="bgr24")      
        return av.VideoFrame.from_ndarray(image, format="bgr24")

st.sidebar.title('Major Project Batch B294')
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Demo'])

if app_mode == 'Home':
    st.title('About Our Project')
elif app_mode == 'Demo':
    st.header('Real-Time Hand Gesture Recognition Using Mediapipe & LSTM')
    st.markdown('To start detecting your ASL gesture click on the "START" button')
    ctx = webrtc_streamer(
    key="example",
    video_processor_factory=OpenCamera,
    rtc_configuration={ # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
