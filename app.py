import streamlit as st
import tensorflow as tf
from skimage import transform
from streamlit_option_menu import option_menu

#def load_model():
Malaria_model=tf.keras.models.load_model('model.h5')
Brain_model=tf.keras.models.load_model('model_vgg19.h5')
 # return model
#with st.spinner('Model is being loaded..'):
 # model=load_model()

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Classification System',
                          
                          ['Malaria Classification',
                           'Brain Tumor Classification'],
                          default_index=0)
    
if (selected == 'Malaria Classification'):
    st.write("""
             # Malaria Detection Using CNN
             """
             )

    file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
    import cv2
    from PIL import Image, ImageOps
    import numpy as np
    st.set_option('deprecation.showfileUploaderEncoding', False)
    def import_and_predict(image_data, Malaria_model):
        
            size = (224,224)    
            image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
            image = np.asarray(image)
            np_image = transform.resize(image, (224, 224, 3))
            np_image = np.expand_dims(np_image, axis=0)
            img = np_image

            prediction = Malaria_model.predict(np_image)
            
            return prediction
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, Malaria_model)
        class_names = ['Detected', 'Not detected']
        string = "Malaria : "+class_names[np.argmax(predictions)]
        st.success(string)
    
if (selected == 'Brain Tumor Classification'):
    st.write("""
             # Brain Tumor Detection Using CNN
             """
             )

    file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
    import cv2
    from PIL import Image, ImageOps
    import numpy as np
    st.set_option('deprecation.showfileUploaderEncoding', False)
    def import_and_predict(image_data, Brain_model):
        
            size = (224,224)    
            image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
            image = np.asarray(image)
            np_image = transform.resize(image, (224, 224, 3))
            np_image = np.expand_dims(np_image, axis=0)
            img = np_image

            prediction = Brain_model.predict(np_image)
            
            return prediction
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, Brain_model)
        class_names = ['Detected', 'Not detected']
        string = "Brain Tumor : "+class_names[np.argmax(predictions)]
        st.success(string)