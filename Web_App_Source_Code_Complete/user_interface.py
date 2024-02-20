import gradio as gr
from utils import *

# Gradio Interface

def main():
  
    io = gr.Interface(
        fn=predict,
        inputs=[
            gr.File(label="Upload an image of a Baby aged between 0 to 5 years", file_types=["image"]),
            gr.Textbox(label="Enter your name:", default=""),  # Added input for the user's name
            gr.Radio(label="Select gender:", choices=["Male", "Female"]),
            gr.Radio(label="Select age unit:", choices=["Years", "Months"]),
            gr.Number(label="Enter age (Only the Number for the Selected Age Unit):", default=1, min=0, max=5),  # Default value for age (range 0 to 5 years)   
        ],
        outputs=[gr.Image(label="Uploaded Image", width=400, height=400),
                 gr.Textbox(label="Malnutrition & Deficiencies Report", show_copy_button = True),
                 gr.Textbox(label="Common Symptoms Of Different Types Of Deficiencies", default = "", multiline = True)
                 ], 
        allow_flagging="manual",
        flagging_options=["Save"],
        title="PREDICTION OF INFANT MALNUTRITION USING CONVOLUTIONAL NEURAL NETWORKS AND IMAGE PROCESSING",
        description="Predict the Malnutrition Type, Severity of Malnutrition, Body Measurements, Vitamin B7, Iron, Vitamin A and Iron Deficiencies",
        theme=gr.themes.Soft()
    )
    io.launch(share=True)

if __name__ == "__main__":
    main()