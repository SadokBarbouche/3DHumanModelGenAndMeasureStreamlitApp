import streamlit as st
import webbrowser
import pyautogui
import time
def generate_url(sex=0, bust=90.4, underbust=80.6, waist=80.2, hip=98.3, neckgirth=33.4, insideleg=76.3, shoulder=36.6, bodyheight=188.0):
    url = 'https://sadokbarbouche.github.io/3Dhumvis/?'
    url += 'sex=' + str(sex)
    url += '&Bust=' + str(bust)
    url += '&UnderBust=' + str(underbust)
    url += '&Waist=' + str(waist)
    url += '&Hip=' + str(hip)
    url += '&NeckGirth=' + str(neckgirth)
    url += '&InsideLeg=' + str(insideleg)
    url += '&Shoulder=' + str(shoulder)
    url += '&BodyHeight=' + str(bodyheight)
    return url


def generate_model(url):
    webbrowser.open_new_tab(url)
    time.sleep(2)
    pyautogui.hotkey('ctrl', 'w')