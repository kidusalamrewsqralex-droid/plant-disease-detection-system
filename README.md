# Green Hand 🌱

**Green Hand** is a web app that helps farmers and plant enthusiasts **detect plant diseases**,**predict crop yields** and **by recommending perfect crops/plants** using AI. It provides actionable insights to improve plant health and optimize farming decisions.  

## Features
- **Plant Disease Detection:** Upload a plant image to instantly identify diseases using a model trained on the **TensorFlow Plant Village dataset**.  
- **Crop Yield Prediction:** Enter farm and crop details to estimate potential yield.  
- **Crop Recommender:** Input soil parameters, weather conditions, and rainfall to receive a **suggested crop** using machine learning models like **Random Forest** and **XGBoost**. Helps farmers make data-driven planting decisions and reduce crop failure.  
- **User-Friendly Interface:** Built with Streamlit for an intuitive and interactive experience.  
- **Data-Driven Insights:** Helps users make informed decisions for better crop management.  

## Technology
- **Frontend:** Streamlit  
- **Machine Learning:** TensorFlow for disease detection, Scikit-learn for yield prediction  
- **Dataset:** TensorFlow Plant Village dataset,kaggle crop yield dataset and kaggle crop recommendation data set in addition to the locally gathered data
- **Deployment:** Streamlit Cloud  

## How to Run Locally
1. Make sure all project files (including `app.py`, `requirements.txt`, and model files) are in the same folder.  
2. Create and activate a virtual environment:  
  run this on your command prompt: python -m venv venv
   # On macOS/Linux
   run code: source venv/bin/activate
   # On Windows -> command prompt(cmd)
   run code: venv\Scripts\activate
3. Install dependencies:
run code: pip install -r requirements.txt
4. Run the app:
run code: streamlit run app.py
5.Open the URL provided in the terminal (usually http://localhost:8501) in a web browser to use the app.

#### Here is the link to my web-app(to access online):https://plant-disease-detection-system-amhcwzukw24svryvcveckt.streamlit.app/
NOTICE:when you try to access the app online,if it says that the app has been put to a sleep,click the "yes wake my app" button.
after running the app:login as admin,access the app and even manage logins.
username:admin
password:admin123

**i would like to ask you to see the screenshots and every element of this file to better understand the app and also use the link above to see the online app,because there will be so many updates and added models after i submit this file.please**


License: MIT License