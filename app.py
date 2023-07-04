import pyvista as pv
import streamlit as st
from stpyvista import stpyvista
from model_generator import *
from measures import *
from streamlit.components.v1 import components
import numpy as np
import requests
from streamlit_lottie import st_lottie
import json
import globals

def load_lottiefile(filepath: str):
    with open(filepath, "r",encoding="utf-8") as f:
        return json.load(f)




def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    


def get_uploaded_file_path(uploaded_file):
    with open(f"tmp/{uploaded_file.name}", "wb") as file:
        file.write(uploaded_file.getbuffer())
    return f"tmp/{uploaded_file.name}"



def show_model(path):
    pv.set_jupyter_backend('panel')
    pv.global_theme.show_scalar_bar = False
    if "pv_model" in st.session_state:
        del st.session_state["pv_model"]    
    plotter = pv.Plotter(window_size=[600, 600])
    mesh = pv.read(path)
    plotter.add_mesh(mesh, color='white', smooth_shading=True)
    camera_position = [0, -25, 500]
    camera_viewup = [0, 0, 1]
    plotter.camera_position = camera_position
    plotter.camera_viewup = camera_viewup
    plotter.background_color = '#0e1117'
    stpyvista(plotter, key="pv_model")
    plotter.clear()    
    
def main():
    st.set_page_config(initial_sidebar_state="collapsed")
    
    st.markdown(
        """
        <style>
        .fullscreen-text {
            top: 0;
            left: 0;
            width: 100%;
            height: 75vh;
            display: flex;
            justify-content: center;
            flex-direction: column;
            align-items: center;
            font-size: 5rem;
            font-weight: bold;

        }
        .fullscreen-text > p {
            font-size: 1.25rem;
            align-items: center;
            text-align: center;
            color: #a8a5ac
        }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    


    st.markdown('<div class="fullscreen-text">AI Tailor<p><br>AI Tailor web service is now available for integration, allowing you to easily incorporate our AI-powered body measurements into your app or platform. With our service, you can provide your users with accurate and convenient measurements of their height, waist, hip, and chest circumference and much more.</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        lottie_coding = load_lottiefile("./assets/tailor.json") 
        lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_coy8mzqf.json")

        
        
        st_lottie(
        lottie_hello,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        height=None,
        width=None,
        key=None,
        
        )


    
    hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    


    lottie_coding = load_lottiefile("./assets/clothes3d.json") 
    lottie_hello = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_gn0tojcq.json")

    
    
    st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
    height=None,
    width=None,
    key=None,
    )
    
    
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    

    st.title("3D Human Model Generator")
    st.info('This is a demo of the alpha version of the 3D Human Model Generator. It is pretty \nslow but it is a proof of concept. The final version will be much faster and will \nhave more features.', icon="🚨")
    st.text('\n')
    
    st.text('\n')
    

    
    col1, col2 = st.columns(2)
    with col1:
        with st.form("valid_input_form"):
            front = st.file_uploader("Upload the front image of your body", type=[
                                    'png', 'jpg', 'jpeg'])
            left = st.file_uploader("Upload the left side image of your body", type=[
                                    'png', 'jpg', 'jpeg'])
            
            model_input = []
            test1 , test2 = False, False
            gender = st.selectbox("Sex", ["Female", "Male"])
            globals.shoulder_width = st.text_input("Real shoulder width (in cm)")
            verify = st.form_submit_button("Generate my 3D model (.obj)")
            sex, bust, underbust, waist, hip, neckgirth, insideleg, shoulder, bodyheight = 0, 0, 0, 0, 0, 0, 0, 0, 0
            if verify:
                if front is None or left is None:
                    st.write("Please upload both images")
                else:
                    front_body, front_infos = process(
                        image=get_uploaded_file_path(front))
                    sex = 0 if gender == "Female" else 1
                    if front_infos['is_backward'] == False and front_infos['orientation'][0] == "NO ORIENTATION" and front_infos['is_valid'] == True and front_infos['pose'] == "A Pose":
                        test1 = True
                        st.write("Front image is valid ✅")
                        front_expander = st.expander("See more front image infos:")
                        front_expander.write(front_infos)
                        model_input = [gender]+list(front_infos["real_body_measurements"].values())
                        print(model_input)
                        sex, bust, underbust, waist, hip, neckgirth, insideleg, shoulder, bodyheight = model_input
                                 
                    else:
                        st.write(
                            "Front image is not valid , please reupload a valid image")

                    left_body, left_infos = process(image=get_uploaded_file_path(left))

                    if left_infos['orientation'][0] == 'LEFT' and left_infos['is_valid']:
                        test2 = True
                        st.write("Left side image is valid ✅")
                        left_expander = st.expander("See more left image infos:")
                        left_expander.write(left_infos)
                    else:
                        st.write(
                            "Left side image is not valid , please reupload a valid image")
                    

    with col2:
        lottie_coding = load_lottiefile("./assets/form.json") 
        lottie_hello = load_lottieurl("https://assets7.lottiefiles.com/private_files/lf30_dmituz7c.json")

        st_lottie(
        lottie_hello,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        height=450,
        width=None,
        key=None,
        )
        

    st.title("3D Human Model Generator : manual input")
    
    col1,col2 = st.columns(2)
    
    with col1:
        lottie_coding = load_lottiefile("./assets/measures.json") 
        lottie_hello = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_EwlEHt.json")        
        st_lottie(
        lottie_hello,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        height=800,
        width=None,
        key=None,
        )
        
        
    with col2:
        with st.form("my_form"):
            sex = st.number_input("Sex", min_value=0,max_value=1, value=0)
            bust = st.number_input("Bust", min_value=79.0,
                                max_value=113.0, value=90.4)
            underbust = st.number_input(
                "Underbust", min_value=70.0, max_value=101.0, value=80.6)
            waist = st.number_input("Waist", min_value=52.0,
                                    max_value=113.0, value=80.2)
            hip = st.number_input("Hip", min_value=79.0,
                                max_value=121.0, value=98.3)
            neckgirth = st.number_input(
                "Neck Girth", min_value=29.0, max_value=45.0, value=33.4)
            insideleg = st.number_input(
                "Inside Leg", min_value=65.0, max_value=95.0, value=76.3)
            shoulder = st.number_input(
                "Shoulder", min_value=29.0, max_value=60.0, value=36.6)
            bodyheight = st.number_input(
                "Body Height", min_value=145.0, max_value=201.0, value=168.0)
            submitted = st.form_submit_button("Generate my 3D model (.obj)")
            if submitted:
                print(bodyheight)
                url = generate_url(sex, bust, underbust, waist, hip, neckgirth, insideleg, shoulder, bodyheight)
                print(url)
                generate_model(url)

    st.title("Visualize my 3D model")
    st.warning('''
    1️⃣ Download the .obj file generated from the previous step or simply upload yours\n
    2️⃣ Upload it and wait for a second\n
    3️⃣ Bingo !
    ''')

    st.sidebar.title(
        "𝜶 Version of the 3D Human Model Generator/ Measurement Tool"
        )
    st.sidebar.markdown(
        "Created by: [@Sadok Barbouche](https://sadokbarbouche.github.io/myPortfolio)"
        )
    st.sidebar.markdown(
         "[👨‍💻 Source code:](https://github.com/SadokBarbouche/3DHumanModelGenAndMeasureStreamlitApp)"
        )
    st.sidebar.info('We are working on making the input from the camera !', icon="⚠️")



if __name__ == "__main__":
    main()
