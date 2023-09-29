import os
import face_recognition
import cv2
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

# Start an MLflow experiment
mlflow.set_tracking_uri("file:/C:/Users/NageshV/Desktop/Projects/live-face-detection/mlruns")
mlflow.start_run()

# Load pre-trained face recognition model
known_face_encodings = []
known_face_names = []

# Directory where your image files are stored
image_directory = "C:/Users/NageshV/Desktop/Projects/live-face-detection\images"

# Loop through the image files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg"):
        # Extract the person's name from the filename
        person_name = os.path.splitext(filename)[0]

        # Load the face image
        image_path = os.path.join(image_directory, filename)
        image = face_recognition.load_image_file(image_path)

        # Attempt to find and encode the face in the image
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) > 0:
            # Assuming one face per image, use the first encoding
            encoding = face_encodings[0]

            # Append the encoding and name to the respective lists
            known_face_encodings.append(encoding)
            known_face_names.append(person_name)
        else:
            print(f"No face detected in {filename}. Skipping.")

# Initialize the camera
video_capture = cv2.VideoCapture(0)

# Initialize the recognized_names list
recognized_names = []

# Create a function to perform face recognition
def recognize_face(frame):
    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    names = []
    # Loop through detected faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default to "Unknown" if no match is found

        # If a match is found, use the name of the known person
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        names.append((top, right, bottom, left, name))

    return face_locations, names  # Return both face locations and recognized names

# Open an MLflow run
with mlflow.start_run(nested=True):
    mlflow.log_params({"image_directory": image_directory})

    while True:
        # Capture a frame from the camera
        ret, frame = video_capture.read()

        # Perform face recognition
        face_locations, recognized_names_batch = recognize_face(frame)

        # Extend the recognized_names list with the batch
        recognized_names.extend(recognized_names_batch)

        # Draw a box and label on the face
        for (top, right, bottom, left, name) in recognized_names_batch:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display the resulting frame
        cv2.imshow("Face Recognition", frame)

        # Exit the loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Log the recognized_names parameter outside the loop
mlflow.log_params({"recognized_names": recognized_names})


# After processing and obtaining recognized names
recognized_names_str = ", ".join(name for (_, _, _, _, name) in recognized_names)

# Save recognized names to a text file as an artifact
with open("recognized_names.txt", "w") as file:
    file.write(recognized_names_str)

# Log the path to the artifact
mlflow.log_artifact("recognized_names.txt")

# Release the camera and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
mlflow.end_run()

