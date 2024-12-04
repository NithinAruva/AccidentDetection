import cv2
import numpy as np
from tkinter import Tk, Label, Button, Text, END
from tkinter.filedialog import askopenfilename
import winsound
import os
import pickle
from tensorflow.keras.models import model_from_json

# Initialize the main window
main = Tk()
main.title("Accident Detection")
main.geometry("1300x1200")

global filename
global classifier

def beep():
    frequency = 2500  
    duration = 1000 
    winsound.Beep(frequency, duration)

names = ['Accident Occured', 'No Accident Occured']

def webcamPredict():
    global classifier
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
            classifier.load_weights("model/model_weights.weights.h5")
            classifier.make_predict_function()   
    videofile = askopenfilename(initialdir="videos")
    if videofile:
        video = cv2.VideoCapture(videofile)
        while video.isOpened():
            ret,frame = video.read()
            if ret:
                # Preprocess the frame
                img = cv2.resize(frame, (120, 120))
                img = img.astype('float32') / 255.0  # Normalize pixel values
                img = np.expand_dims(img, axis=0)  # Add batch dimension
                
                # Predict using the model
                predict = classifier.predict(img)
                predicted_class = np.argmax(predict)
                result = names[predicted_class]
                
                # Display prediction
                cv2.putText(frame, result, (500, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("video frame", frame)
                
                # Play beep sound if predicted class is 0
                if predicted_class == 0:
                    beep()
                
                # Exit condition
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            else:
                break
        
        # Release video capture and close all windows
        video.release()
        cv2.destroyAllWindows()

# Configure title
font = ('times', 16, 'bold')
title = Label(main, text='Accident Detection System', anchor='center', justify='center')
title.config(bg='yellow4', fg='white', font=font)  
title.config(height=3, width=120)       
title.pack(side='top', fill='x')  # Center the title at the top

# Configure the upload button
font1 = ('times', 13, 'bold')
predictButton = Button(main, text="Upload Video & Detect Accident", command=webcamPredict, font=font1)
predictButton.place(relx=0.5, rely=0.5, anchor='center')  # Center the button in the window

# Configure the background
main.config(bg='magenta3')

# Run the GUI
main.mainloop()
