import cv2
import mediapipe as mp 
import math 
import os

def calculate_2d_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def eye_width_averages_cms():
    eye_width_averages_all_ages_cms = {
        # From the below given age groups, difference in eye width between consecutive age groups has been taken as 0.2 cms
        "0-3 months": 1.6, # average eye width in cms for age 0 months to 3 months (including 0 month but excluding 3 months)
        "3-6 months": 2.0, # average eye width in cms for age 3 months to 6 months (including 3 months but excluding 6 months), similar for others below
        "6-12 months": 2.4,
        "12-24 months": 2.6,
        "24-36 months": 2.8,
        "36-72 months": 3.0,
        # From the below given age groups, difference in eye width between consecutive age groups has been taken as 0.3 cms
        "72-120 months": 3.2,
        "120-156 months": 3.5,
        "156-216 months": 3.8,
        "216-1800 months": 4.1
    }
    return eye_width_averages_all_ages_cms

def age_in_months(age_unit, age_number):
    # Obtaining the user's age in months 
    user_age_months = 0
    if age_unit.lower() == "months":
        user_age_months = age_number
    elif age_unit.lower() == "years":
        user_age_months = age_number * 12
    return user_age_months

def eye_width_cms(user_age_months, eye_width_averages_all_ages_cms):
    # Conditional ladder to get the average eye width of a person belonging to a particular age group (months)
    if 0 <= user_age_months and user_age_months < 3: # 0 months to 3 months (including 0 month but excluding 3 months)
        user_eye_width_cms = eye_width_averages_all_ages_cms['0-3 months']
    elif 3 <= user_age_months and user_age_months < 6: # 3 months to 6 months (including 3 months but excluding 6 months), similar for others below
        user_eye_width_cms = eye_width_averages_all_ages_cms['3-6 months']
    elif 6 <= user_age_months and user_age_months < 12:
        user_eye_width_cms = eye_width_averages_all_ages_cms['6-12 months']
    elif 12 <= user_age_months and user_age_months < 24:
        user_eye_width_cms = eye_width_averages_all_ages_cms['12-24 months']   
    elif 24 <= user_age_months and user_age_months < 36:
        user_eye_width_cms = eye_width_averages_all_ages_cms['24-36 months']  
    elif 36 <= user_age_months and user_age_months < 72:
        user_eye_width_cms = eye_width_averages_all_ages_cms['36-72 months']  
    elif 72 <= user_age_months and user_age_months < 120:
        user_eye_width_cms = eye_width_averages_all_ages_cms['72-120 months'] 
    elif 120 <= user_age_months and user_age_months < 156:
        user_eye_width_cms = eye_width_averages_all_ages_cms['120-156 months'] 
    elif 156 <= user_age_months and user_age_months < 216:
        user_eye_width_cms = eye_width_averages_all_ages_cms['156-216 months'] 
    elif 216 <= user_age_months and user_age_months < 1800:
        user_eye_width_cms = eye_width_averages_all_ages_cms['216-1800 months'] 
    return user_eye_width_cms

def load_holistic():
    import mediapipe as mp 
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
    return holistic

def set_drawing_style():
    import mediapipe as mp 
    mp_drawing = mp.solutions.drawing_utils
    drawing_style_for_lms = mp_drawing.DrawingSpec(color = (0,255,0), thickness = 2, circle_radius = 2)
    drawing_style_for_connections_of_lms = mp_drawing.DrawingSpec(color = (255,0,0), thickness = 2, circle_radius = 2)
    return drawing_style_for_lms, drawing_style_for_connections_of_lms

def obtain_all_landmarks(image_path, holistic_model): 
    # Getting the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Converting the loaded image to RGB to make the detections in the image using mediapipe holistic
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Getting the coordinates of the landmarks for the image using holistic
    all_body_landmarks = holistic_model.process(image_RGB)

    # Converting the RGB image back to it's original BGR format after getting the coordinates of all the landmarks using mediapipe holistic, We'll draw all the landmarks detected on this original BGR form of the image
    image_BGR = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)

    return image_BGR, all_body_landmarks

def draw_face_landmarks_and_connections(image_BGR, all_body_landmarks, drawing_style_for_lms, drawing_style_for_connections_of_lms):
    mp_drawing = mp.solutions.drawing_utils 
    mp_holistic = mp.solutions.holistic
    face_landmarks = all_body_landmarks.face_landmarks
    mp_drawing.draw_landmarks(image_BGR, face_landmarks, mp_holistic.FACEMESH_TESSELATION, drawing_style_for_lms, drawing_style_for_connections_of_lms)
    return face_landmarks

def draw_right_hand_landmarks_and_connections(image_BGR, all_body_landmarks, drawing_style_for_lms, drawing_style_for_connections_of_lms):
    mp_drawing = mp.solutions.drawing_utils 
    mp_holistic = mp.solutions.holistic
    right_hand_landmarks = all_body_landmarks.right_hand_landmarks
    mp_drawing.draw_landmarks(image_BGR, right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, drawing_style_for_lms, drawing_style_for_connections_of_lms)
    return right_hand_landmarks

def draw_left_hand_landmarks_and_connections(image_BGR, all_body_landmarks, drawing_style_for_lms, drawing_style_for_connections_of_lms):
    mp_drawing = mp.solutions.drawing_utils 
    mp_holistic = mp.solutions.holistic
    left_hand_landmarks = all_body_landmarks.left_hand_landmarks
    mp_drawing.draw_landmarks(image_BGR, left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, drawing_style_for_lms, drawing_style_for_connections_of_lms)
    return left_hand_landmarks

def draw_pose_landmarks_and_connections(image_BGR, all_body_landmarks, drawing_style_for_lms, drawing_style_for_connections_of_lms):
    mp_drawing = mp.solutions.drawing_utils 
    mp_holistic = mp.solutions.holistic
    pose_landmarks = all_body_landmarks.pose_landmarks
    mp_drawing.draw_landmarks(image_BGR, pose_landmarks, mp_holistic.POSE_CONNECTIONS, drawing_style_for_lms, drawing_style_for_connections_of_lms)
    return pose_landmarks

def image_height_width_px(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_height_px, img_width_px, channels = image.shape
    return img_height_px, img_width_px

def draw_circle_on_point(image, point_coords_list):
    cv2.circle(image, point_coords_list, 2, (0, 0, 255), thickness=cv2.FILLED)

def draw_line_joining_points(image, point1_coords, point2_coords):
    # Drawing a line joining the Right end & Left end landmarks of the Right eye
    cv2.line(image, point1_coords, point2_coords, (0, 255, 0), 3) 

def eye_width_in_px(face_landmarks, img_width_px, img_height_px):
    right_end_right_eye_lm_coords = face_landmarks.landmark[130]
    right_end_right_eye_lm_coord_x, right_end_right_eye_lm_coord_y, right_end_right_eye_lm_coord_z = right_end_right_eye_lm_coords.x, right_end_right_eye_lm_coords.y, right_end_right_eye_lm_coords.z
    right_end_right_eye_lm_coord_x_px = int(right_end_right_eye_lm_coord_x * img_width_px)
    right_end_right_eye_lm_coord_y_px = int(right_end_right_eye_lm_coord_y * img_height_px)
    right_end_right_eye_lm_coords_only_x_and_y = [right_end_right_eye_lm_coord_x_px, right_end_right_eye_lm_coord_y_px] 

    left_end_right_eye_lm_coords = face_landmarks.landmark[243]
    left_end_right_eye_lm_coord_x, left_end_right_eye_lm_coord_y, left_end_right_eye_lm_coord_z = left_end_right_eye_lm_coords.x, left_end_right_eye_lm_coords.y, left_end_right_eye_lm_coords.z
    left_end_right_eye_lm_coord_x_px = int(left_end_right_eye_lm_coord_x * img_width_px)
    left_end_right_eye_lm_coord_y_px = int(left_end_right_eye_lm_coord_y * img_height_px)
    left_end_right_eye_lm_coords_only_x_and_y = [left_end_right_eye_lm_coord_x_px, left_end_right_eye_lm_coord_y_px]

    user_eye_width_px = calculate_2d_distance(right_end_right_eye_lm_coords_only_x_and_y, left_end_right_eye_lm_coords_only_x_and_y) # calculating eye width in pixels (distance between 2 landmarks in pixels) using custom function for 2D space

    return user_eye_width_px

def px_to_cm_conversion_factor(eye_width_in_cms, eye_width_in_px):
    px_to_cm_conversion_factor = eye_width_in_cms / eye_width_in_px
    return px_to_cm_conversion_factor

def head_diameter_cms(pose_landmarks, img_width_px, img_height_px, px_to_cm_conversion_factor):
    right_end_face_coords = pose_landmarks.landmark[8]  # Index 8 corresponds to the right ear
    right_end_face_coord_x, right_end_face_coord_y, right_end_face_coord_z = right_end_face_coords.x, right_end_face_coords.y, right_end_face_coords.z
    right_end_face_coord_x_px, right_end_face_coord_y_px = int(right_end_face_coord_x * img_width_px), int(right_end_face_coord_y * img_height_px)
    right_end_face_lm_coords_only_x_and_y = [right_end_face_coord_x_px, right_end_face_coord_y_px]

    left_end_face_coords = pose_landmarks.landmark[7]  # Index 7 corresponds to the left ear
    left_end_face_coord_x, left_end_face_coord_y, left_end_face_coord_z = left_end_face_coords.x, left_end_face_coords.y, left_end_face_coords.z
    left_end_face_coord_x_px, left_end_face_coord_y_px = int(left_end_face_coord_x * img_width_px), int(left_end_face_coord_y * img_height_px)
    left_end_face_lm_coords_only_x_and_y = [left_end_face_coord_x_px, left_end_face_coord_y_px]

    user_head_diameter_px = calculate_2d_distance(right_end_face_lm_coords_only_x_and_y, left_end_face_lm_coords_only_x_and_y) 

    user_head_diameter_cms = user_head_diameter_px * px_to_cm_conversion_factor
    return user_head_diameter_cms

def head_circumference_cms(user_head_diameter_cms):
    user_head_circumference_cms = math.pi * user_head_diameter_cms
    return user_head_circumference_cms

def arm_length_cms(pose_landmarks, img_width_px, img_height_px, px_to_cm_conversion_factor):
    left_shoulder_coords = pose_landmarks.landmark[11]  # Index 11 corresponds to the left shoulder
    left_shoulder_coord_x, left_shoulder_coord_y, left_shoulder_coord_z = left_shoulder_coords.x, left_shoulder_coords.y, left_shoulder_coords.z
    left_shoulder_coord_x_px, left_shoulder_coord_y_px = int(left_shoulder_coord_x * img_width_px), int(left_shoulder_coord_y * img_height_px)
    left_shoulder_lm_coords_only_x_and_y = [left_shoulder_coord_x_px, left_shoulder_coord_y_px]

    left_wrist_coords = pose_landmarks.landmark[15]     # Index 15 corresponds to the left wrist
    left_wrist_coord_x, left_wrist_coord_y, left_wrist_coord_z = left_wrist_coords.x, left_wrist_coords.y, left_wrist_coords.z
    left_wrist_coord_x_px, left_wrist_coord_y_px = int(left_wrist_coord_x * img_width_px), int(left_wrist_coord_y * img_height_px)
    left_wrist_lm_coords_only_x_and_y = [left_wrist_coord_x_px, left_wrist_coord_y_px]

    user_arm_length_px = calculate_2d_distance(left_shoulder_lm_coords_only_x_and_y, left_wrist_lm_coords_only_x_and_y) 
    user_arm_length_cms = user_arm_length_px * px_to_cm_conversion_factor
    return user_arm_length_cms

def height_cms(face_landmarks, pose_landmarks, img_width_px, img_height_px, px_to_cm_conversion_factor):
    
    head_top_coords = face_landmarks.landmark[10] # Index 10 corresponds to the top of the head
    left_ankle_coords = pose_landmarks.landmark[27] # Index 27 corresponds to the left ankle
    right_ankle_coords = pose_landmarks.landmark[28] # Index 28 corresponds to the right ankle

    # Obtaining the x & y coordinates in standard values of the landmark of the Top of the Head
    head_top_coord_x, head_top_coord_y, head_top_coord_z = head_top_coords.x, head_top_coords.y, head_top_coords.z
    head_top_coord_x_px, head_top_coord_y_px = int(head_top_coord_x * img_width_px), int(head_top_coord_y * img_height_px)
    head_top_lm_coords_only_x_and_y = [head_top_coord_x_px, head_top_coord_y_px]

    # Obtaining the x & y coordinates in standard values of the landmark of the Right Ankle
    right_ankle_coord_x, right_ankle_coord_y, right_ankle_coord_z = right_ankle_coords.x, right_ankle_coords.y, right_ankle_coords.z
    right_ankle_coord_x_px, right_ankle_coord_y_px = int(right_ankle_coord_x * img_width_px), int(right_ankle_coord_y * img_height_px)
    right_ankle_lm_coords_only_x_and_y = [right_ankle_coord_x_px, right_ankle_coord_y_px]

    # Obtaining the x & y coordinates in standard values of the landmark of the Left Ankle
    left_ankle_coord_x, left_ankle_coord_y, left_ankle_coord_z = left_ankle_coords.x, left_ankle_coords.y, left_ankle_coords.z
    left_ankle_coord_x_px, left_ankle_coord_y_px = int(left_ankle_coord_x * img_width_px), int(left_ankle_coord_y * img_height_px)
    left_ankle_lm_coords_only_x_and_y = [left_ankle_coord_x_px, left_ankle_coord_y_px]

    # Calculating the distance between top of the head and right ankle in pixels 
    distance_head_top_right_ankle_px = calculate_2d_distance(head_top_lm_coords_only_x_and_y, right_ankle_lm_coords_only_x_and_y) 

    # Calculating the distance between top of the head and right ankle in centimetres using pixel to centimetre conversion factor
    distance_head_top_right_ankle_cms = distance_head_top_right_ankle_px * px_to_cm_conversion_factor

    # Calculating the distance between top of the head and left ankle in pixels 
    distance_head_top_left_ankle_px = calculate_2d_distance(head_top_lm_coords_only_x_and_y, left_ankle_lm_coords_only_x_and_y) 

    # Calculating the distance between top of the head and left ankle in centimetres using pixel to centimetre conversion factor
    distance_head_top_left_ankle_cms = distance_head_top_left_ankle_px * px_to_cm_conversion_factor

    user_height_cms = (distance_head_top_right_ankle_cms + distance_head_top_left_ankle_cms) / 2

    return user_height_cms

def close_holistic(holistic):
    holistic.close()

def resize_image(image, fx, fy):
    image_BGR_resized = cv2.resize(image, (0, 0), fx = fx, fy = fy)
    return image_BGR_resized

def image_window(window_name, image, key_to_close_window_a_to_z):
    # Create a window with a specific name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Resize the window to a specific width and height
    cv2.resizeWindow(window_name, 800, 600)
    # Display the image in the resized window
    cv2.imshow(window_name, image)
    k = cv2.waitKey(0)
    if k == ord(key_to_close_window_a_to_z):
        cv2.destroyAllWindows()

def create_report(file_path, user_name, user_gender, user_age_in_months, user_head_circumference_cms, user_arm_length_cms, user_height_cms):
    # Text file for storing all the calculated measurements of the body
    report_file_path = file_path # Path where the Child's body measurements report has to be saved 
    # Check if the file already exists (It could be the Body measurements report of the previous user)
    if os.path.exists(report_file_path):
        # Delete the existing body measurements report
        os.remove(report_file_path)
        print(f"Existing Body Measurements Report : '{report_file_path}' deleted.")
    # Create a New Body measurements report file for the new user 
    report_file =  open(report_file_path, 'w')
    print(f"New Body Measurements Report at '{report_file_path}' created and opened in write mode.")
    report_file.write("User Body Measurements Report\n\n\n")
    report_file.write(f"User's Name : {user_name} \n\n")
    report_file.write(f"User's Gender : {user_gender} \n\n")
    report_file.write(f"User's Age In Months : {user_age_in_months} \n\n")
    report_file.write(f"User's Head Circumference in Centimetres : {user_head_circumference_cms} \n\n")
    report_file.write(f"User's Arm Length in Centimetres : {user_arm_length_cms} \n\n")
    report_file.write(f"User's Height in Centimetres : {user_height_cms} \n\n")
