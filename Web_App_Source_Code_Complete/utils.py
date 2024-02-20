import body_measurements_module as bm
from numpy import *
import tensorflow as tf
from keras.models import load_model
from datetime import datetime
import pandas as pd

user_name = ""
user_age_in_months = 0
user_gender = ""
exam_date = ""
user_height_in_cms = 0
user_head_circumference_in_cms = 0
user_arm_length_in_cms = 0
malnutrition_report = ""
malnutrition_desc = ""
CNN_models_report = ""

# ___________________________Code for determining the Severity of Malnutrition____________________________

# Paths of all the Excel files to be used
head_circumference_for_age_boys = "C:\\Users\\OMOLP094\\Desktop\\Complete Infant Malnutrition Prediction Project\\malnutrition_severity_datasets\\head_circumference_for_age_boys_0_to_5_years.xlsx"

head_circumference_for_age_girls = "C:\\Users\\OMOLP094\\Desktop\\Complete Infant Malnutrition Prediction Project\\malnutrition_severity_datasets\\head_circumference_for_age_girls_0_to_5_years.xlsx"

height_for_age_boys = "C:\\Users\\OMOLP094\\Desktop\\Complete Infant Malnutrition Prediction Project\\malnutrition_severity_datasets\\height_for_age_boys_0_to_5_years.xlsx"

height_for_age_girls = "C:\\Users\\OMOLP094\\Desktop\\Complete Infant Malnutrition Prediction Project\\malnutrition_severity_datasets\\height_for_age_girls_0_to_5_years.xlsx"

# _______________________________________________________________________________________________________

# Load all the Excel files into a Pandas DataFrame 
head_circumference_for_age_boys_df = pd.read_excel(head_circumference_for_age_boys, sheet_name='head_circumference_for_boys', header=0, usecols=['Month', 'M', 'SD'])

# Renaming the columns
head_circumference_for_age_boys_df_updated = head_circumference_for_age_boys_df.rename(columns={'Month': 'Age In Months', 'M': 'Median', 'SD': 'Standard Deviation'})

# _______________________________________________________________________________________________________

head_circumference_for_age_girls_df = pd.read_excel(head_circumference_for_age_girls, sheet_name='head_circumference_for_girls', header=0, usecols=['Month', 'M', 'SD'])

head_circumference_for_age_girls_df_updated = head_circumference_for_age_girls_df.rename(columns={'Month': 'Age In Months', 'M': 'Median', 'SD': 'Standard Deviation'})

# _______________________________________________________________________________________________________

height_for_age_boys_df = pd.read_excel(height_for_age_boys, sheet_name='height_for_age_boys', header=0, usecols=['Month', 'M', 'SD'])

height_for_age_boys_df_updated = height_for_age_boys_df.rename(columns={'Month': 'Age In Months', 'M': 'Median', 'SD': 'Standard Deviation'})

# _______________________________________________________________________________________________________

height_for_age_girls_df = pd.read_excel(height_for_age_girls, sheet_name='height_for_age_girls', header=0, usecols=['Month', 'M', 'SD'])

height_for_age_girls_df_updated = height_for_age_girls_df.rename(columns={'Month': 'Age In Months', 'M': 'Median', 'SD': 'Standard Deviation'})

# _______________________________________________________________________________________________________

# sheet_name='Sheet1': This parameter specifies the name of the sheet in the Excel file that you want to read. In this case, it is set to 'Sheet1'. If your Excel file contains multiple sheets, you can use this parameter to specify the sheet you're interested in.

# header=0: The header parameter is used to indicate which row in the Excel file should be considered as the header, i.e., the row containing column names. In this case, it is set to 0, which means that the first row (index 0) of the specified sheet will be treated as the header. This is where the column names are expected.

# usecols=['Column1', 'Column2']: The usecols parameter allows you to specify which columns from the Excel file you want to read into the DataFrame. In this example, it is set to a list containing 'Column1' and 'Column2'. Only these two columns will be extracted and included in the resulting DataFrame.

# _______________________________________________________________________________________________________


median_height = 0
standard_deviation_height = 0
median_head_circumference = 0
standard_deviation_head_circumference = 0

# Calculating Z Scores : 

def calculate_z_score(calculated_value, median_value, standard_deviation):
    z_score = (calculated_value - median_value) / standard_deviation
    return z_score

def interpret_z_score(z_score):
    if -1 < z_score < 0:
        category = "Normal (Well Nourished)"
    elif -2 < z_score < -1:
        category = "Marginally Stunted (Mildly Malnourished)"
    elif -3 < z_score < -2:
        category = "Moderately Stunted (Moderately Malnourished)"
    elif z_score < -3:
        category = "Severely Stunted (Severely Malnourished)"
    else:
        category = "Unknown category"

    return category

def median_and_standard_deviation_height(user_gender, user_age_months, height_for_age_boys_dataframe, height_for_age_girls_dataframe):
    global median_height, standard_deviation_height
    if user_gender == "male": 
        # Age value in months for which you want to get the values
        age_value = user_age_months 

        # Filter the DataFrame based on the age value
        filtered_height_df = height_for_age_boys_dataframe[height_for_age_boys_dataframe['Age In Months'] == age_value]

        # Check if there are any rows matching the age value
        if not filtered_height_df.empty:
            # Get the values from 'Median' and 'Standard Deviation' columns
            median_height = filtered_height_df['Median'].iloc[0]
            standard_deviation_height = filtered_height_df['Standard Deviation'].iloc[0]

    elif user_gender == "female":
        # Age value in months for which you want to get the values
        age_value = user_age_months 

        # Filter the DataFrame based on the age value
        filtered_height_df = height_for_age_girls_dataframe[height_for_age_girls_dataframe['Age In Months'] == age_value]

        # Check if there are any rows matching the age value
        if not filtered_height_df.empty:
            # Get the values from 'M' and 'SD' columns
            median_height = filtered_height_df['Median'].iloc[0]
            standard_deviation_height = filtered_height_df['Standard Deviation'].iloc[0]

    return median_height, standard_deviation_height
            

def median_and_standard_deviation_head_circumference(user_gender, user_age_months, head_circumference_for_age_boys_dataframe, head_circumference_for_age_girls_dataframe):
    global median_head_circumference, standard_deviation_head_circumference
    if user_gender == "male": 
        # Age value in months for which you want to get the values
        age_value = user_age_months 

        # Filter the DataFrame based on the age value
        filtered_head_circumference_df = head_circumference_for_age_boys_dataframe[head_circumference_for_age_boys_dataframe['Age In Months'] == age_value]

        # Check if there are any rows matching the age value
        if not filtered_head_circumference_df.empty:
            # Get the values from 'Median' and 'Standard Deviation' columns
            median_head_circumference = filtered_head_circumference_df['Median'].iloc[0]
            standard_deviation_head_circumference = filtered_head_circumference_df['Standard Deviation'].iloc[0]

    elif user_gender == "female":
        # Age value in months for which you want to get the values
        age_value = user_age_months 

        # Filter the DataFrame based on the age value
        filtered_head_circumference_df = head_circumference_for_age_girls_dataframe[head_circumference_for_age_girls_dataframe['Age In Months'] == age_value]

        # Check if there are any rows matching the age value
        if not filtered_head_circumference_df.empty:
            # Get the values from 'M' and 'SD' columns
            median_head_circumference = filtered_head_circumference_df['Median'].iloc[0]
            standard_deviation_head_circumference = filtered_head_circumference_df['Standard Deviation'].iloc[0]

    return median_head_circumference, standard_deviation_head_circumference

# Labels Dictionaries for CNN Models

malnutrition_type_labels_dict = {
    "healthy" : 0,
    "kwashiorkor" : 1,
    "marasmus" : 2
}

dry_skin_vitamin_b7_iron_deficiency_labels_dict = {
    'healthy_skin' : 0,
    'dry_skin' : 1
}

vitamin_a_deficiency_labels_dict = {
    "healthy_eyes" : 0,
    "vitamin_a_deficiency" : 1
}

iodine_deficiency_labels_dict = {
    "no_iodine_deficiency" : 0,
    "iodine_deficiency" : 1
}

# Function to make predictions, this function would be called inside the predict function (i.e, function that is being called by the Gradio UI) & it would be used for making predictions for the CNN model (Predictor) chosen on the Gradio UI for the Uploaded image by the user

# Scores for predicted class only
def prediction(uploaded_image_path, loaded_model, model_labels_dict):
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(uploaded_image_path, target_size=(180, 180))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = expand_dims(image_array, axis=0)
    image_scaled = image_array / 255.0

    # Make predictions using the loaded model
    predictions = loaded_model.predict(image_scaled)

    # Apply softmax to obtain probabilities
    probabilities = tf.nn.softmax(predictions).numpy()

    # Get the predicted class index
    predicted_class_index = argmax(predictions)

    # Display the predicted class name
    predicted_class = "Unknown"
    for class_name, class_num_label in model_labels_dict.items():
        if class_num_label == predicted_class_index:
            predicted_class = class_name

    # Get the probability and confidence score for the predicted class
    probability = probabilities[0, predicted_class_index]
    confidence_score = probability * 100

    # Create a string with class name, probability, and confidence score
    result_string = f"Predicted Class: {predicted_class}, Probability: {probability:.4f}, Confidence Score: {confidence_score:.2f}%"

    return image, predicted_class, result_string

""" # Scores for all classes 
def prediction(uploaded_image_path, loaded_model, model_labels_dict):
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(uploaded_image_path, target_size=(180, 180))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = expand_dims(image_array, axis=0)
    image_scaled = image_array / 255.0

    # Make predictions using the loaded model
    predictions = loaded_model.predict(image_scaled)

    # Apply softmax to obtain probabilities
    probabilities = tf.nn.softmax(predictions).numpy()

    # Get the predicted class index
    predicted_class_index = argmax(predictions)

    # Display the predicted class name
    predicted_class = "Unknown"
    for class_name, class_num_label in model_labels_dict.items():
        if class_num_label == predicted_class_index:
            predicted_class = class_name

    # Create a string with class name, probability, and confidence score for all classes
    result_string = f"Predicted Class: {predicted_class}\n"
    
    for class_name, probability in zip(model_labels_dict.keys(), probabilities.flatten()):
        confidence_score = probability * 100
        result_string += f"{class_name}: Probability: {probability:.4f}, Confidence Score: {confidence_score:.2f}%\n"

    return image, predicted_class, result_string """

# Storing Paths of CNN models in Variables

malnutrition_type_model = "C:\\Users\\OMOLP094\\Desktop\\Infant_Malnutrition\\Gradio_UI_for_Malnutrition_and_Deficiencies_using_CNNs\\CNN_Models_for_detecting_Malnutrition_and_Deficiencies\\malnutrition_type_enhanced_model\\malnutrition_type_classification_enhanced_CNN_model.h5"

skin_condition_model = "C:\\Users\\OMOLP094\\Desktop\\Infant_Malnutrition\\Gradio_UI_for_Malnutrition_and_Deficiencies_using_CNNs\\CNN_Models_for_detecting_Malnutrition_and_Deficiencies\\dry_skin_classification_enhanced_model\\dry_and_healthy_skin_classification_enhanced_CNN_model.h5"

eye_condition_model = "C:\\Users\\OMOLP094\\Desktop\\Infant_Malnutrition\\Gradio_UI_for_Malnutrition_and_Deficiencies_using_CNNs\\CNN_Models_for_detecting_Malnutrition_and_Deficiencies\\vitamin_A_deficiency_enhanced_model\\vitamin_A_deficient_eyes_classification_enhanced_CNN_model.h5"

iodine_deficiency_model = "C:\\Users\\OMOLP094\\Desktop\\Infant_Malnutrition\\Gradio_UI_for_Malnutrition_and_Deficiencies_using_CNNs\\CNN_Models_for_detecting_Malnutrition_and_Deficiencies\\iodine_deficiency_enhanced_model\\healthy_iodine_deficient_classification_enhanced_CNN_model.h5"

# List containing Paths of CNN models & their Labels Dictionaries

model_classes_list = [[malnutrition_type_model, malnutrition_type_labels_dict],
                      [skin_condition_model, dry_skin_vitamin_b7_iron_deficiency_labels_dict],
                      [eye_condition_model, vitamin_a_deficiency_labels_dict],
                      [iodine_deficiency_model, iodine_deficiency_labels_dict]]

# Function to be called by the Gradio UI

def predict(uploaded_image, user_name, user_gender, user_age_unit, user_age_number):
    malnutrition_report = "Infant Malnutrition Diagnosis Report \n\n SECTION 1 (Patient Details) \n"
    global CNN_models_report

    # Get the current date
    current_date = datetime.now().date()
    # Print or use the current date
    date_string = f"Current Date: {str(current_date)}"

    malnutrition_report = f"{malnutrition_report} \n Name : {user_name} \n Age : {str(user_age_number)} {user_age_unit} \n Gender : {user_gender} \n Date of Examination: {date_string} \n\n"

    uploaded_image_path = uploaded_image.name

    holistic_model = bm.load_holistic()

    drawing_style_lms, drawing_style_lms_connections = bm.set_drawing_style()

    image_BGR, all_body_lms = bm.obtain_all_landmarks(uploaded_image_path, holistic_model)

    face_landmarks = bm.draw_face_landmarks_and_connections(image_BGR, all_body_lms, drawing_style_lms, drawing_style_lms_connections)

    right_hand_landmarks = bm.draw_right_hand_landmarks_and_connections(image_BGR, all_body_lms, drawing_style_lms, drawing_style_lms_connections)

    left_hand_landmarks = bm.draw_left_hand_landmarks_and_connections(image_BGR, all_body_lms, drawing_style_lms, drawing_style_lms_connections)

    pose_landmarks = bm.draw_pose_landmarks_and_connections(image_BGR, all_body_lms, drawing_style_lms, drawing_style_lms_connections)

    eye_width_averages_cms_dict = bm.eye_width_averages_cms()

    user_age_in_months = bm.age_in_months(user_age_unit, user_age_number)

    user_eye_width_cms = bm.eye_width_cms(user_age_in_months, eye_width_averages_cms_dict)

    img_height_px, img_width_px = bm.image_height_width_px(uploaded_image_path)

    user_eye_width_px = bm.eye_width_in_px(face_landmarks, img_width_px, img_height_px)

    px_to_cm_conversion_factor = bm.px_to_cm_conversion_factor(user_eye_width_cms, user_eye_width_px)

    user_head_diameter_in_cms = bm.head_diameter_cms(pose_landmarks, img_width_px, img_height_px, px_to_cm_conversion_factor)

    user_head_circumference_in_cms = bm.head_circumference_cms(user_head_diameter_in_cms)

    user_arm_length_in_cms = bm.arm_length_cms(pose_landmarks, img_width_px, img_height_px, px_to_cm_conversion_factor)

    user_height_in_cms = bm.height_cms(face_landmarks, pose_landmarks, img_width_px, img_height_px, px_to_cm_conversion_factor)

    bm.close_holistic(holistic_model)
    
    malnutrition_report = malnutrition_report + f"SECTION 2 (Body Measurements) \n\n Height : {round(user_height_in_cms, 2)} cms \n Head circumference : {round(user_head_circumference_in_cms, 2)} cms \n Arm Length : {round(user_arm_length_in_cms, 2)} cms \n\n"

    median_height, SD_height = median_and_standard_deviation_height(user_gender.lower(), user_age_in_months, height_for_age_boys_df_updated, height_for_age_girls_df_updated)
    height_z_score = calculate_z_score(user_height_in_cms, median_height, SD_height)
    severity_height = interpret_z_score(height_z_score)
    percent_stunting = (user_height_in_cms / median_height) * 100

    median_head_circum, SD_head_circum = median_and_standard_deviation_head_circumference(user_gender.lower(), user_age_in_months, head_circumference_for_age_boys_df_updated, head_circumference_for_age_girls_df_updated)
    head_circum_z_score = calculate_z_score(40, median_head_circum, SD_head_circum)
    severity_head_circum = interpret_z_score(head_circum_z_score)

    malnutrition_report = malnutrition_report + f"SECTION 3 (Severity Assessment) - Malnutrition Type (Physical) : Stunting \n\n Malnutrition Severity Scale : Grade 1 To 4 \n\n Height for Age Z Score : {height_z_score} \n Malnutrition Severity (Based On Height For Age Z Score) : {severity_height} \n\n Head Circumference for Age Z Score : {head_circum_z_score} \n Malnutrition Severity (Based On Head Circumference For Age Z Score) : {severity_head_circum} \n Percent Stunting : {percent_stunting}\n\n"

    malnutrition_report = malnutrition_report +  "SECTION 4 (Type of Malnutrition and Deficiencies) \n"

    for model in model_classes_list:
        model_path = model[0]
        model_labels_dict = model[1]
        loaded_model = load_model(model_path)
        # Call the prediction function
        uploaded_image, predicted_class, result_string = prediction(uploaded_image_path, loaded_model, model_labels_dict)
        # predicted_class = predicted_class.replace("_", " ") # removing all the underscores from the malnutrition report string as it contains the names of predicted classes which have "_" in them
        malnutrition_report = malnutrition_report + "\n" + result_string 
        CNN_models_report = CNN_models_report + result_string + "\n"
        
    malnutrition_report = malnutrition_report.title() # Converting the malnutrition report string into title case

    malnutrition_desc = "Useful Information : \n\nKwashiorkor (protein malnutrition predominant) \n\nMarasmus (deficiency in calorie intake) \n\nKwashiorkor : Children with kwashiorkor may have a normal weight or even be overweight, but the weight is not distributed normally due to edema. The height-for-age may be relatively preserved. \n\nMarasmus : Characterized by low weight-for-age, low weight-for-height, and low height-for-age. Severe wasting is a key anthropometric feature.\n\nCommon Symptoms of Vitamin A Deficiency : dry eyes, dry skin, frequent infections, inability to see in dim light, or spots in the eyeball. \n\nCommon Symptoms of Vitamin B7 Deficiency : hair loss, skin rashes, brittle nails, fatigue, muscle pain, and neurological symptoms such as depression, lethargy, and tingling in the extremities. \n\nCommon Symptoms of Iron Deficiency : \n\nWhole body : dizziness, fatigue, or light-headedness \n\nHeart : fast heart rate or palpitations \n\nOther Common Symptoms : brittle nails, pallor, or shortness of breath \n\nCommon Symptoms of Iodine Deficiency : Swelling of the thyroid glands in the neck, a visible lump (goiter) on the neck, Weight gain, fatigue, and weakness, Thinning of hair, Dry skin"
    
    return uploaded_image, malnutrition_report, malnutrition_desc



