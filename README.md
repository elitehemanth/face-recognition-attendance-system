# 🫆 face-recognition-attendance-system
A face-recognition attendance system is a small marvel of applied computer vision. It watches for a human face through a webcam, captures the frame, and transforms that image into a compact numerical “embedding”—a kind of mathematical fingerprint of the person’s features. A pre-trained deep-learning model such as VGG-Face, loaded through libraries like DeepFace, provides the recognition engine. Its learned weights, stored in the vgg_face_weights.h5 file, allow the system to compare the captured face with enrolled users and decide who’s present.

Once the face is matched, the system quietly records attendance: check-ins, check-outs, timestamps, and status updates. Behind the scenes it keeps a history file—often JSON or a small database—that grows into a chronological record of every visit. A friendly UI, sometimes built with tools like Streamlit or Flask, turns this machinery into a simple dashboard where you can register new users, view logs, and watch the recognition process in real time.

The overall idea is to replace manual sign-ins with a lightweight, automated system that recognizes people almost instantly. You end up with something that feels both futuristic and slightly magical: walk in, look at the camera for a heartbeat, and the machine quietly knows who you are.
<hr>

# 🏃‍➡️🏃‍♂️‍➡️Run facescan.py
<br>
streamlit run facescan.py
<br>
<br>
On the first execution, the system automatically downloads vgg_face_weights.h5 and stores it at:
<br>


C:\Users\username\\.deepface\weights\vgg_face_weights.h5


# 🤖 install vgg_face_weights.h5
If the file does not download automatically, follow the steps below:

Download the weights manually using the link:
https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5

Place the downloaded file in the directory:

C:\Users\username\\.deepface\weights\vgg_face_weights.h5


This ensures the system can access the required model weights during execution.


# NOT MY STYLE
