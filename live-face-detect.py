import os
import face_recognition
import cv2

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

while True:
    # Capture a frame from the camera
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through detected faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with known face encodingsmkd
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"  # Default to "Unknown" if no match is found

        # If a match is found, use the name of the known person
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box and label on the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Exit the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
