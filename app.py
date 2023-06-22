import pyvista as pv
import streamlit as st
from stpyvista import stpyvista
from model_generator import *
from measures import *
from streamlit.components.v1 import components
import numpy as np


def get_uploaded_file_path(uploaded_file):
    with open(f"tmp/{uploaded_file.name}", "wb") as file:
        file.write(uploaded_file.getbuffer())    
    return f"tmp/{uploaded_file.name}"

def main():

    st.title("Valid Input Checker")
    st.text("This is a demo of the alpha version of the 3D Human Model Generator. It is pretty \nslow but it is a proof of concept. The final version will be much faster and will \nhave more features.")
    st.text('\n')
    with st.form("valid_input_form"):    
        front = st.file_uploader("Upload the front image of your body", type=['png', 'jpg', 'jpeg'])
        left = st.file_uploader("Upload the left side image of your body", type=['png', 'jpg', 'jpeg'])
        verify = st.form_submit_button("Submit")
        if verify:

            front_body,front_infos = process(image=get_uploaded_file_path(front))

            if front_infos['is_backward'] == False and front_infos['orientation'][0] == "NO ORIENTATION" and front_infos['is_valid']==True and front_infos['pose']=="A Pose":
                st.write("Front image is valid")
            else:
                st.write("Front image is not valid , please reupload a valid image")

            left_body,left_infos = process(image=get_uploaded_file_path(left))

            if left_infos['orientation'][0] == 'LEFT' and left_infos['is_valid']:
                st.write("Left side image is valid")
            else:
                st.write("Left side image is not valid , please reupload a valid image")
            
    st.title("3D Human Model Generator")
    with st.form("my_form"):
        sex = st.selectbox("Sex", [0, 1])
        bust = st.number_input("Bust", min_value=79.0, max_value=113.0, value=90.4)
        underbust = st.number_input("Underbust", min_value=70.0, max_value=101.0, value=80.6)
        waist = st.number_input("Waist", min_value=52.0, max_value=113.0, value=80.2)
        hip = st.number_input("Hip", min_value=79.0, max_value=121.0, value=98.3)
        neckgirth = st.number_input("Neck Girth", min_value=29.0, max_value=45.0, value=33.4)
        insideleg = st.number_input("Inside Leg", min_value=65.0, max_value=95.0, value=76.3)
        shoulder = st.number_input("Shoulder", min_value=29.0, max_value=60.0, value=36.6)
        bodyheight = st.number_input("Body Height", min_value=145.0, max_value=201.0, value=168.0)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Sex:", sex)
            st.write("Bust:", bust)
            st.write("Underbust:", underbust)
            st.write("Waist:", waist)
            st.write("Hip:", hip)
            st.write("Neck Girth:", neckgirth)
            st.write("Inside Leg:", insideleg)
            st.write("Shoulder:", shoulder)
            st.write("Body Height:", bodyheight)
            pv.global_theme.show_scalar_bar = False
            plotter = pv.Plotter(window_size=[600, 600])

            mesh = pv.read(generate_model(sex, bust, underbust, waist, hip, neckgirth, insideleg, shoulder, bodyheight))
            plotter.add_mesh(mesh, color='white', smooth_shading=True)
            camera_position = [0, -25, 500] 
            camera_viewup = [0, 1, 1]  
            plotter.camera_position = camera_position
            plotter.camera_viewup = camera_viewup
            plotter.background_color = '#0e1117'
            stpyvista(plotter, key="pv_model")
    
    st.sidebar.title("Alpha Version of the 3D Human Model Generator/ Measurement Tool")


if __name__ == "__main__":
    main()