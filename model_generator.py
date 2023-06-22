import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

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


@st.cache
def generate_model(sex, bust, underbust, waist, hip, neckgirth, insideleg, shoulder, bodyheight):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    driver = webdriver.Chrome(options=chrome_options)
    url = generate_url(sex=sex, bust=bust, underbust=underbust, waist=waist, hip=hip, neckgirth=neckgirth, insideleg=insideleg, shoulder=shoulder, bodyheight=bodyheight)
    driver.get(url)
    item_value = driver.execute_script('return window.localStorage.getItem("exportedObj");')
    file_path = "tmp/exported_obj.obj"
    with open(file_path, "w") as file:
        if item_value is not None:
            file.write(item_value)
    driver.quit()
    return file_path