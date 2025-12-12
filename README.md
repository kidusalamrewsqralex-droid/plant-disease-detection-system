# Green Hand 🌱

**Green Hand** is a web app that helps farmers and plant enthusiasts **detect plant diseases** and **predict crop yields** using AI. It provides actionable insights to improve plant health and optimize farming decisions.  

## Features
- **Plant Disease Detection:** Upload a plant image to instantly identify diseases using a model trained on the **TensorFlow Plant Village dataset**.  
- **Crop Yield Prediction:** Enter farm and crop details to estimate potential yield.  
- **User-Friendly Interface:** Built with Streamlit for an intuitive and interactive experience.  
- **Data-Driven Insights:** Helps users make informed decisions for better crop management.  

## Technology
- **Frontend:** Streamlit  
- **Machine Learning:** TensorFlow for disease detection, Scikit-learn for yield prediction  
- **Dataset:** TensorFlow Plant Village dataset  
- **Deployment:** Streamlit Cloud  



#### Here is the link to my web-app:https://plant-disease-detection-system-amhcwzukw24svryvcveckt.streamlit.app/




## How to Run Locally
1. 1. Make sure all project files (including `app.py`, `requirements.txt`, and model files) are in the same folder.  
2. Create and activate a virtual environment:  
  run this on your command prompt: python -m venv venv
   # On macOS/Linux
   run code: source venv/bin/activate
   # On Windows
   run code: venv\Scripts\activate
3. Install dependencies:
run code: pip install -r requirements.txt
4. Run the app:
run code: streamlit run app.py
5.Open the URL provided in the terminal (usually http://localhost:8501) in a web browser to use the app.



License: MIT License