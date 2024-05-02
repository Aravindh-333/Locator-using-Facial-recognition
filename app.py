from flask import Flask, render_template, request, redirect, url_for
import cv2
import face_recognition as fr
import time
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'first_face_file' not in request.files:
        return "No file part"
    
    file = request.files['first_face_file']

    if file.filename == '':
        return "No selected file"

    if file:
        try:
            faceOne = fr.load_image_file(file)
            RgbFaceOne = cv2.cvtColor(faceOne, cv2.COLOR_BGR2RGB)
            faceOneEnco = fr.face_encodings(RgbFaceOne)[0]

            cap = cv2.VideoCapture(0)

            match_found = False

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                RgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                faceLocs = fr.face_locations(RgbFrame)
                
                if not faceLocs:
                    continue

                for faceLoc in faceLocs:
                    faceEncodings = fr.face_encodings(RgbFrame, [faceLoc])
                    faceTwoEnco = faceEncodings[0]

                    results = fr.compare_faces([faceOneEnco], faceTwoEnco)

                    if results[0]:
                        print("Face Matched")

                        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

                        top, right, bottom, left = faceLoc
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, "Matched at: " + current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                        ret, jpeg = cv2.imencode('.jpg', frame)
                        matched_image = jpeg.tobytes()

                        match_found = True
                        break  

                if match_found:
                    cap.release()
                    cv2.destroyAllWindows()
                    return render_template('matched_image.html', matched_image=matched_image)
        except Exception as e:
            return f"An error occurred: {e}"

    return "Error processing the file"

if __name__ == '__main__':
    app.run(debug=True)
