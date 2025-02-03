import os
import cv2
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from collections import defaultdict as dd
from deepface import DeepFace
import tensorflow as tf
import warnings

# Suppress TensorFlow and other warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs (INFO/WARNING)
tf.get_logger().setLevel('ERROR')  # Set TensorFlow logger level to ERROR
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress deprecation warnings

# Set memory growth for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Parameters
email = '<YOUR_EMAIL>'
password = '<YOUR_PASSWORD>'
send_to_email = '<RECIPIENT_EMAIL>'
subject = '******EMOTION RECOGNITION******'
message = '*****RESULTS****'
stats_file = 'Statistics.txt'

EMOTIONS = ["angry", "fear", "neutral", "sad", "disgust", "happy", "surprise"]
emotion_counts = {emotion: 0 for emotion in EMOTIONS}

# Start video stream
cv2.namedWindow('Emotion Predictor')
camera = cv2.VideoCapture(0)
start_time = time.time()
flag = 0

# Write stats to file
f = open(stats_file, "w+")
f.truncate(0)

while flag == 0:
    # Capture frame from camera
    frame = camera.read()[1]
    frame = cv2.resize(frame, (300, 300))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV
    face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process faces
    for (fX, fY, fW, fH) in faces:
        # Crop face
        face = frame[fY:fY + fH, fX:fX + fW]

        # Use DeepFace to analyze emotions
        try:
            analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            
            # Handle multiple faces or single-face output
            if isinstance(analysis, list):
                analysis = analysis[0]  # Get the first face's result
            
            dominant_emotion = analysis['dominant_emotion']
            emotion_counts[dominant_emotion] += 1
            print(f"Detected emotion: {dominant_emotion}")

            # Draw bounding box and label
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
            cv2.putText(frame, dominant_emotion, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print(f"Emotion analysis failed: {e}")

    # Display results
    cv2.imshow("Emotion Predictor", frame)

    # Check if emotion count exceeds threshold for sending email
    end_time = time.time()
    if end_time - start_time > 4:
        if emotion_counts["sad"] > 1000:
            # Save statistics
            d = dd(int)
            for emotion, count in emotion_counts.items():
                d[emotion] = count
            with open("emotions.txt", "w+") as f2:
                f2.truncate(0)
                f2.write(f"Emotions: \n{str(d)}")

            # Send email
            msg = MIMEMultipart()
            msg['From'] = email
            msg['To'] = send_to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(message, 'plain'))

            # Attach emotion statistics file
            filename = os.path.basename("emotions.txt")
            attachment = open("emotions.txt", "rb")
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename={filename}")
            msg.attach(part)

            # Send email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(email, password)
            text = msg.as_string()
            server.sendmail(email, send_to_email, text)
            server.quit()

            print("Email sent!")
            flag = 1

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        f.close()
        break

f.close()
camera.release()
cv2.destroyAllWindows()
