import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Đồ án tốt nghiệp")
st.header("Machine Learning")

menu=["Home","Capstone Project"]
choice=st.sidebar.selectbox('Menu',menu)

if choice=='Home':
    st.subheader("[Trang chủ](https://csc.edu.vn/)")
elif choice=='Capstone Project':
    st.subheader("[Đồ án tốt nghiệp Machine Learning](https://csc.edu.vn/data-science-machine-learning/Data-Science-Certificate_199)")
    st.write("""Có 3 chủ đề:
    
    -Topic 1: Tran Cong Thien
    -Topic 2: Phan Van A
    -Topic 3: Nguyễn Đức C
    
    """)