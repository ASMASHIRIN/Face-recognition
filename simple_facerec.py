import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
        # resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        # """Loads the known faces from image files into memory."""
        images_path = glob.glob(os.path.join(images_path,"*.*"))
        print("{} encoding images found.".format(len(images_path)))

        # store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # get the filname only from the initial file path 
            basename = os.path.basename(img_path)
            (filename,ext) = os.path.splitext(basename)
            # get Encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # store file name and file encoding
            self.known_face_names.append(filename)
            self.known_face_encodings.append(img_encoding)
    print("Encoding images loaded")


    def detetct_known_faces(self,frame):
        small_frame = cv2.resize(frame, (0,0), fx=self.frame_resizing, fy=self.frame_resizing)
        # find all the faces and face encodings in the current frame of vedio
        # convert the image from BGR color (which openCV uses) to RGB color(which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # see if the face os a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding) 
            name = "Unknown"

            # if a match was found in known_face_encodings , just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]


            # or instead, use the known face with the smallest distance to the new face 
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

            # convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations/ self.frame_resizing
        return face_locations.astype(int), face_names