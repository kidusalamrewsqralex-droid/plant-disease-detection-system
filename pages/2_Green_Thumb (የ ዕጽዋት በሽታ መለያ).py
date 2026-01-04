# pages/Plant_Disease_Detection.py
import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
from auth import require_login
import tensorflow as tf
require_login()
st.title("🌱 GREEN THUMB")
tab1,tab2=st.tabs(["CNN MobileNetV2 model","CNN model"])
class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew',
        'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___healthy',
        'Corn___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
        'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy',
        'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
        'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
    ]

    # -----------------------------
    # Disease responses (example)
    # -----------------------------
disease_responses = {
    "Apple___Apple_scab": """
                        Diagnosis: Apple Scab is a fungal disease caused by *Venturia inaequalis*.
                        \nAmharic (ምርመራ): የፖም ስካብ በ Venturia inaequalis ፈንገስ አማካኝነት የሚመጣ በሽታ ነው።
                        \nCause: High humidity and wet conditions promote fungal spore growth.
                        \nAmharic (መንስኤ): ከፍተኛ እርጥበት እና መርጠብ ለፈንገስ ስፖሮች እድገት ምቹ ሁኔታ ይፈጥራል።
                        \nTreatment: Apply fungicides such as captan or mancozeb during early stages. Remove fallen leaves and infected fruit.
                        \nAmharic (ህክምና): በሽታው እንደጀመረ ካፕታን (captan) ወይም ማንኮዜብ (mancozeb) መጠቀም። የወደቁ ቅጠሎችን እና ፍሬዎችን ማስወገድ።
                        \nPrevention: Ensure proper pruning for airflow, avoid overhead watering, and clear debris around the tree base.
                        \nAmharic (መከላከያ): ለአየር ዝውውር ሲባል ተክሉን መከርከም፣ በቅጠሉ ላይ ውሃ አለማፍሰስ እና የዛፉን ስር ማጽዳት።
                        """,
    "Apple___Black_rot": """
                        Diagnosis: Black Rot is caused by the fungus *Botryosphaeria obtusa*.
                        \nAmharic (ምርመራ): ጥቁር ብስባሽ (Black Rot) በ Botryosphaeria obtusa ፈንገስ አማካኝነት ይከሰታል።
                        \nCause: Wet, warm weather and infected pruning wounds.
                        \nAmharic (መንስኤ): እርጥብ እና ሞቃታማ የአየር ሁኔታ እንዲሁም በተቆረጡ የተክል አካላት ላይ የሚፈጠር ቁስል ነው።
                        \nTreatment: Prune out cankers and apply fungicide during growing season.
                        \nAmharic (ህክምና): የቆሰሉ አካላትን መቁረጥ እና በዕድገት ወቅት የፈንገስ ማጥፊያ መጠቀም።
                        \nPrevention: Sanitize tools, remove mummified fruits, and improve tree spacing.
                        \nAmharic (መከላከያ): የመስሪያ መሳሪያዎችን ማጽዳት፣ የደረቁ ፍሬዎችን ማስወገድ እና የተክሎችን ርቀት መጠበቅ።
                        """,
    "Apple___Cedar_apple_rust": """
                        Diagnosis: Cedar Apple Rust is a fungal disease linked to both apple and cedar trees.
                        \nAmharic (ምርመራ): የፖም ዝገት (Cedar Apple Rust) ከፖም እና ከሴዳር ዛፎች ጋር የተያያዘ የፈንገስ በሽታ ነው።
                        \nCause: Caused by *Gymnosporangium juniperi-virginianae*, spreads between cedar and apple trees.
                        \nAmharic (መንስኤ): በ Gymnosporangium juniperi-virginianae የሚመጣ ሲሆን በሁለቱ ዛፎች መካከል ይተላለፋል።
                        \nTreatment: Apply fungicides during early growth stages. Remove nearby cedar trees if possible.
                        \nAmharic (ህክምና): በዕድገት መጀመሪያ ላይ የፈንገስ ማጥፊያ መጠቀም። በአቅራቢያ ያሉ የሴዳር ዛፎችን ማስወገድ።
                        \nPrevention: Use rust-resistant varieties and avoid planting near cedars.
                        \nAmharic (መከላከያ): ዝገትን የሚቋቋሙ ዝርያዎችን መጠቀም እና በሴዳር ዛፎች አቅራቢያ አለመትከል።
                        """,
    "Apple___healthy": "✅ The apple plant is healthy. (ፖሙ ጤናማ ነው።) Continue proper watering, pruning, and disease monitoring.",
    "Blueberry___healthy": "✅ The blueberry plant is healthy. (ብሉቤሪው ጤናማ ነው።) Maintain well-drained, acidic soil.",
    "Cherry___healthy": "✅ The cherry plant is healthy. (ቼሪው ጤናማ ነው።) Monitor for signs of mildew or rot during humid seasons.",
    "Cherry___Powdery_mildew": """
                        Diagnosis: Powdery mildew is a fungal infection that forms a white powder on leaves.
                        \nAmharic (ምርመራ): ዋግ (Powdery mildew) በቅጠሎች ላይ ነጭ ዱቄት መሰል ምልክት የሚያሳይ የፈንገስ በሽታ ነው።
                        \nCause: High humidity, poor air circulation.
                        \nAmharic (መንስኤ): ከፍተኛ እርጥበት እና ደካማ የአየር ዝውውር።
                        \nTreatment: Apply sulfur-based or neem oil sprays.
                        \nAmharic (ህክምና): ሰልፈር ያላቸው መድሃኒቶችን ወይም የኒም ኦይል (neem oil) መጠቀም።
                        \nPrevention: Prune regularly and avoid watering late in the day.
                        \nAmharic (መከላከያ): አዘውትሮ መከርከም እና ምሽት ላይ ውሃ አለማፍሰስ።
                        """,
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": """
                        Diagnosis: Gray Leaf Spot is caused by *Cercospora zeae-maydis*.
                        \nAmharic (ምርመራ): ግራጫ የቅጠል ነጥብ (Gray Leaf Spot) በ Cercospora zeae-maydis የሚመጣ ነው።
                        \nCause: Warm, humid environments with high leaf moisture.
                        \nAmharic (መንስኤ): ሞቃታማ፣ እርጥበት አዘል ሁኔታ እና የቅጠል መርጠብ።
                        \nTreatment: Use fungicides like strobilurins or triazoles.
                        \nAmharic (ህክምና): ስትሮቢሉሪን ወይም ትሪያዞል ያላቸውን የፈንገስ ማጥፊያዎች መጠቀም።
                        \nPrevention: Rotate crops and select resistant hybrids.
                        \nAmharic (መከላከያ): ሰብልን ማፈራረቅ እና በሽታን የሚቋቋሙ ዝርያዎችን መምረጥ።
                        """,
    "Corn___Common_rust": """
                        Diagnosis: Common Rust is caused by *Puccinia sorghi*.
                        \nAmharic (ምርመራ): ተራ ዝገት (Common Rust) በ Puccinia sorghi የሚከሰት ነው።
                        \nCause: Spread by wind-borne spores under moist conditions.
                        \nAmharic (መንስኤ): በእርጥብ ሁኔታዎች በንፋስ በሚሰራጩ ስፖሮች አማካኝነት ይተላለፋል።
                        \nTreatment: Use fungicides if infection is severe.
                        \nAmharic (ህክምና): በሽታው ከበረታ የፈንገስ ማጥፊያ መጠቀም።
                        \nPrevention: Plant resistant corn varieties.
                        \nAmharic (መከላከያ): ዝገትን የሚቋቋሙ የበቆሎ ዝርያዎችን መትከል።
                        """,
    "Corn___Northern_Leaf_Blight": """
                        Diagnosis: Northern Leaf Blight is caused by *Exserohilum turcicum*, leads to cigar-shaped lesions.
                        \nAmharic (ምርመራ): የሰሜን ቅጠል ብላይት በ Exserohilum turcicum የሚመጣ ሲሆን የሲጋራ ቅርጽ ያላቸው ምልክቶች ይታዩበታል።
                        \nCause: Prolonged wetness and mild temperatures.
                        \nAmharic (መንስኤ): ለረጅም ጊዜ የሚቆይ እርጥበት እና መጠነኛ ሙቀት።
                        \nTreatment: Apply fungicides early in disease cycle.
                        \nAmharic (ህክምና): በሽታው እንደጀመረ የፈንገስ ማጥፊያ መጠቀም።
                        \nPrevention: Rotate crops, use disease-resistant hybrids.
                        \nAmharic (መከላከያ): ሰብልን ማፈራረቅ እና በሽታን የሚቋቋሙ ዝርያዎችን መጠቀም።
                        """,
    "Corn___healthy": "✅ The corn plant is healthy. (በቆሎው ጤናማ ነው።) Monitor for discoloration and maintain fertilizer schedule.",
    "Grape___Black_rot": """
                        Diagnosis: Black rot is a common fungal disease in grapes caused by *Guignardia bidwellii*.
                        \nAmharic (ምርመራ): ጥቁር ብስባሽ በወይን ላይ የሚከሰት እና በ Guignardia bidwellii የሚመጣ በሽታ ነው።
                        \nCause: Wet weather and poor airflow.
                        \nAmharic (መንስኤ): እርጥብ የአየር ሁኔታ እና ደካማ የአየር ዝውውር።
                        \nTreatment: Apply fungicides and remove infected berries and leaves.
                        \nAmharic (ህክምና): የፈንገስ ማጥፊያ መጠቀም እና የተጠቁ ፍሬዎችንና ቅጠሎችን ማስወገድ።
                        \nPrevention: Train vines properly and prune regularly.
                        \nAmharic (መከላከያ): የወይኑን ሃረግ በትክክል መምራት እና አዘውትሮ መከርከም።
                        """,
    "Grape___Esca_(Black_Measles)": """
                        Diagnosis: Esca (Black Measles) is a trunk disease caused by multiple fungi.
                        \nAmharic (ምርመራ): ኢስካ (ጥቁር ኩፍኝ) በብዙ ፈንገሶች አማካኝነት የሚመጣ የግንድ በሽታ ነው።
                        \nCause: Enters through pruning wounds, worsened by drought stress.
                        \nAmharic (መንስኤ): በመከርከሚያ ቁስሎች በኩል የሚገባ ሲሆን በድርቅ ወቅት ይባባሳል።
                        \nTreatment: No cure, remove infected vines.
                        \nAmharic (ህክምና): መድኃኒት የለውም፤ የታመሙትን ሃረጎች ማስወገድ።
                        \nPrevention: Prune carefully and avoid stress to vines.
                        \nAmharic (መከላከያ): በጥንቃቄ መከርከም እና ተክሉ ለጭንቀት (ድርቅ) እንዳይጋለጥ ማድረግ።
                        """,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": """
                        Diagnosis: Leaf Blight causes dark, angular spots on grape leaves.
                        \nAmharic (ምርመራ): የቅጠል ብላይት በወይን ቅጠሎች ላይ ጥቁር እና ማዕዘን ያላቸው ምልክቶችን ያመጣል።
                        \nCause: Caused by *Isariopsis clavispora*, thrives in wet environments.
                        \nAmharic (መንስኤ): በ Isariopsis clavispora የሚመጣ ሲሆን በእርጥብ አካባቢዎች ይስፋፋል።
                        \nTreatment: Use copper fungicides and remove infected leaves.
                        \nAmharic (ህክምና): ኮፐር ያላቸውን የፈንገስ ማጥፊያዎች መጠቀም እና የታመሙ ቅጠሎችን ማስወገድ።
                        \nPrevention: Improve air circulation, reduce overhead irrigation.
                        \nAmharic (መከላከያ): የአየር ዝውውርን ማሻሻል እና በላይ በኩል ውሃ ማጠጣትን መቀነስ።
                        """,
    "Grape___healthy": "✅ The grape plant is healthy. (ወይኑ ጤናማ ነው።) Continue proper training and disease monitoring.",
    "Orange___Haunglongbing_(Citrus_greening)": """
                        Diagnosis: Huanglongbing (HLB), or citrus greening, is caused by a bacterium spread by psyllids.
                        \nAmharic (ምርመራ): የሲትረስ ግሪኒንግ (ቢጫ መሆን) በነፍሳት በሚሰራጭ ባክቴሪያ አማካኝነት የሚመጣ በሽታ ነው።
                        \nCause: Insect vector *Diaphorina citri* transmits the bacteria.
                        \nAmharic (መንስኤ): ዳያፎሪና ሲትሪ የተባሉ ነፍሳት ባክቴሪያውን ያስተላልፋሉ።
                        \nTreatment: No cure. Remove infected trees.
                        \nAmharic (ህክምና): መድኃኒት የለውም፤ የታመሙ ዛፎችን መንቀል።
                        \nPrevention: Control psyllids and plant resistant rootstocks.
                        \nAmharic (መከላከያ): ነፍሳቱን መቆጣጠር እና በሽታን የሚቋቋሙ ዝርያዎችን መትከል።
                        """,
    "Peach___Bacterial_spot": """
                        Diagnosis: Bacterial spot causes lesions on leaves and fruit.
                        \nAmharic (ምርመራ): ባክቴሪያል ስፖት በቅጠሎች እና በፍሬዎች ላይ ቁስሎችን ያስከትላል።
                        \nCause: Caused by *Xanthomonas campestris*, thrives in rainy weather.
                        \nAmharic (መንስኤ): በ Xanthomonas campestris የሚመጣ ሲሆን በዝናባማ ወቅት ይባባሳል።
                        \nTreatment: Copper sprays and resistant cultivars.
                        \nAmharic (ህክምና): የኮፐር ርጭቶችን እና በሽታን የሚቋቋሙ ዝርያዎችን መጠቀም።
                        \nPrevention: Avoid overhead irrigation and prune for airflow.
                        \nAmharic (መከላከያ): በላይ በኩል ውሃ አለማፍሰስ እና ለአየር ዝውውር መከርከም።
                        """,
    "Peach___healthy": "✅ The peach tree is healthy. (ኮኩ ጤናማ ነው።) Monitor during wet seasons for leaf spots.",
    "Pepper,_bell___Bacterial_spot": """
                        Diagnosis: Bacterial spot affects leaves and fruit of bell peppers.
                        \nAmharic (ምርመራ): ባክቴሪያል ስፖት በቃሪያ ቅጠሎች እና ፍሬዎች ላይ ጉዳት ያደርሳል።
                        \nCause: Spread by contaminated tools and wet conditions.
                        \nAmharic (መንስኤ): በተበከሉ መሳሪያዎች እና በእርጥብ ሁኔታዎች ይሰራጫል።
                        \nTreatment: Use copper-based sprays.
                        \nAmharic (ህክምና): ኮፐር ያላቸውን መድሃኒቶች መጠቀም።
                        \nPrevention: Avoid working with wet plants, sanitize tools.
                        \nAmharic (መከላከያ): ተክሎቹ እርጥብ በሆኑበት ወቅት አለመስራት እና መሳሪያዎችን ማጽዳት።
                        """,
    "Pepper,_bell___healthy": "✅ The pepper plant is healthy. (ቃሪያው ጤናማ ነው።) Maintain warm soil and use mulch.",
    "Potato___Early_blight": """
                        Diagnosis: Early blight is caused by *Alternaria solani*.
                        \nAmharic (ምርመራ): ኤርሊ ብላይት በ Alternaria solani የሚከሰት የድንች በሽታ ነው።
                        \nCause: Warm temperatures and humidity.
                        \nAmharic (መንስኤ): ሞቃታማ የሙቀት መጠን እና እርጥበት።
                        \nTreatment: Use chlorothalonil or mancozeb sprays.
                        \nAmharic (ህክምና): ክሎሮታሎኒል ወይም ማንኮዜብ መጠቀም።
                        \nPrevention: Rotate crops, avoid overhead watering.
                        \nAmharic (መከላከያ): ሰብልን ማፈራረቅ እና በላይ በኩል ውሃ አለማፍሰስ።
                        """,
    "Potato___Late_blight": """
                        Diagnosis: Late blight is caused by *Phytophthora infestans*.
                        \nAmharic (ምርመራ): ሌት ብላይት በ Phytophthora infestans የሚመጣ በጣም አደገኛ የድንች በሽታ ነው።
                        \nCause: Cool, wet conditions.
                        \nAmharic (መንስኤ): ቀዝቃዛ እና እርጥብ ሁኔታዎች።
                        \nTreatment: Apply fungicides such as cymoxanil.
                        \nAmharic (ህክምና): ሳይሞክሳኒል (cymoxanil) ያላቸውን መድሃኒቶች መጠቀም።
                        \nPrevention: Remove infected plants immediately and rotate crops.
                        \nAmharic (መከላከያ): የታመሙ ተክሎችን ወዲያውኑ ማስወገድ እና ሰብልን ማፈራረቅ።
                        """,
    "Potato___healthy": "✅ The potato plant is healthy. (ድንቹ ጤናማ ነው።) Hill soil around stems and avoid waterlogging.",
    "Raspberry___healthy": "✅ The raspberry plant is healthy. (ራስቤሪው ጤናማ ነው።) Mulch properly and prune regularly.",
    "Soybean___healthy": "✅ The soybean plant is healthy. (አኩሪ አተሩ ጤናማ ነው።) Check for aphids and fungal symptoms.",
    "Squash___Powdery_mildew": """
                        Diagnosis: Powdery mildew is caused by *Podosphaera xanthii*.
                        \nAmharic (ምርመራ): ዋግ (Powdery mildew) በዱባ ቅጠል ላይ በ Podosphaera xanthii የሚመጣ ነው።
                        \nCause: Dry days followed by humid nights.
                        \nAmharic (መንስኤ): ደረቅ ቀናት እና እርጥብ ምሽቶች።
                        \nTreatment: Apply sulfur or neem oil-based sprays.
                        \nAmharic (ህክምና): ሰልፈር ወይም ኒም ኦይል መጠቀም።
                        \nPrevention: Plant in sunny areas and space properly.
                        \nAmharic (መከላከያ): ፀሐያማ በሆኑ ቦታዎች መትከል እና በቂ ርቀት መስጠት።
                        """,
    "Strawberry___Leaf_scorch": """
                        Diagnosis: Leaf scorch is caused by fungal pathogens.
                        \nAmharic (ምርመራ): የቅጠል መቃጠል በፈንገስ አማካኝነት የሚመጣ በሽታ ነው።
                        \nCause: High humidity and poor air movement.
                        \nAmharic (መንስኤ): ከፍተኛ እርጥበት እና ደካማ የአየር ዝውውር።
                        \nTreatment: Use fungicides and remove infected leaves.
                        \nAmharic (ህክምና): የፈንገስ ማጥፊያ መጠቀም እና የታመሙ ቅጠሎችን ማስወገድ።
                        \nPrevention: Avoid overcrowding and improve drainage.
                        \nAmharic (መከላከያ): ተክሎችን አለማጨናነቅ እና የውሃ ፍሳሽን ማሻሻል።
                        """,
    "Strawberry___healthy": "✅ The strawberry plant is healthy. (እንጆሪው ጤናማ ነው።) Maintain spacing and moist soil.",
    "Tomato___Bacterial_spot": """
                        Diagnosis: Bacterial spot causes black lesions on leaves and fruit.
                        \nAmharic (ምርመራ): ባክቴሪያል ስፖት በቅጠልና ፍሬ ላይ ጥቁር ነጠብጣቦችን ያመጣል።
                        \nCause: Wet, warm conditions.
                        \nAmharic (መንስኤ): እርጥብ እና ሞቃታማ ሁኔታዎች።
                        \nTreatment: Copper-based fungicides.
                        \nAmharic (ህክምና): ኮፐር ያላቸው መድሃኒቶችን መጠቀም።
                        \nPrevention: Use clean seeds and avoid overhead watering.
                        \nAmharic (መከላከያ): ንጹህ ዘር መጠቀም እና በላይ በኩል ውሃ አለማፍሰስ።
                        """,
    "Tomato___Early_blight": """
                        Diagnosis: Early blight is caused by *Alternaria solani*.
                        \nAmharic (ምርመራ): ኤርሊ ብላይት በ Alternaria solani የሚመጣ የቲማቲም በሽታ ነው።
                        \nCause: Poor air circulation and leaf wetness.
                        \nAmharic (መንስኤ): ደካማ የአየር ዝውውር እና የቅጠል መርጠብ።
                        \nTreatment: Use mancozeb or chlorothalonil sprays.
                        \nAmharic (ህክምና): ማንኮዜብ ወይም ክሎሮታሎኒል መጠቀም።
                        \nPrevention: Rotate crops and remove infected debris.
                        \nAmharic (መከላከያ): ሰብልን ማፈራረቅ እና የተጠቁ ቅሪቶችን ማስወገድ።
                        """,
    "Tomato___Late_blight": """
                        Diagnosis: Late blight is caused by *Phytophthora infestans*.
                        \nAmharic (ምርመራ): ሌት ብላይት በ Phytophthora infestans የሚመጣ ነው።
                        \nCause: Cool, moist conditions.
                        \nAmharic (መንስኤ): ቀዝቃዛ እና እርጥብ ሁኔታዎች።
                        \nTreatment: Apply fungicides quickly and remove affected plants.
                        \nAmharic (ህክምና): መድሃኒቶችን በፍጥነት መጠቀም እና የታመሙትን ማስወገድ።
                        \nPrevention: Avoid overhead watering and use resistant varieties.
                        \nAmharic (መከላከያ): በላይ በኩል ውሃ አለማፍሰስ እና በሽታን የሚቋቋሙ ዝርያዎችን መጠቀም።
                        """,
    "Tomato___Leaf_Mold": """
                        Diagnosis: Leaf mold is caused by *Fulvia fulva*.
                        \nAmharic (ምርመራ): የቅጠል አሻጋሪ (Leaf mold) በ Fulvia fulva ይከሰታል።
                        \nCause: High humidity in greenhouses or shaded areas.
                        \nAmharic (መንስኤ): ከፍተኛ እርጥበት (በተለይ ግሪንሃውስ ውስጥ)።
                        \nTreatment: Use fungicides and increase ventilation.
                        \nAmharic (ህክምና): መድሃኒት መጠቀም እና አየር እንዲገባ ማድረግ።
                        \nPrevention: Prune regularly and avoid dense foliage.
                        \nAmharic (መከላከያ): አዘውትሮ መከርከም እና ቅጠሎች እንዳይጨናነቁ ማድረግ።
                        """,
    "Tomato___Septoria_leaf_spot": """
                        Diagnosis: Caused by *Septoria lycopersici*, shows small spots on leaves.
                        \nAmharic (ምርመራ): በ Septoria lycopersici የሚመጣ ሲሆን ትናንሽ ነጥቦችን ያሳያል።
                        \nCause: High humidity, wet foliage.
                        \nAmharic (መንስኤ): ከፍተኛ እርጥበት እና የቅጠሎች መርጠብ።
                        \nTreatment: Use chlorothalonil-based fungicides.
                        \nAmharic (ህክምና): ክሎሮታሎኒል ያላቸው መድሃኒቶችን መጠቀም።
                        \nPrevention: Space plants well and avoid overhead watering.
                        \nAmharic (መከላከያ): በቂ ርቀት መስጠት እና በላይ በኩል ውሃ አለማጠጣት።
                        """,
    "Tomato___Spider_mites Two-spotted_spider_mite": """
                        Diagnosis: Two-spotted spider mite infestation causes stippling and webbing on leaves.
                        \nAmharic (ምርመራ): ባለ ሁለት ነጥብ ማይቶች በቅጠል ላይ ድር እና ነጠብጣቦችን ያመጣሉ።
                        \nCause: Dry conditions, lack of predators.
                        \nAmharic (መንስኤ): ደረቅ ሁኔታዎች እና የተፈጥሮ ጠላቶች (ነፍሳት) አለመኖር።
                        \nTreatment: Use miticides or neem oil.
                        \nAmharic (ህክምና): የማይት ማጥፊያ ወይም ኒም ኦይል መጠቀም።
                        \nPrevention: Maintain moderate humidity, encourage natural predators.
                        \nAmharic (መከላከያ): መጠነኛ እርጥበት መጠበቅ እና ጠቃሚ ነፍሳትን ማበረታታት።
                        """,
    "Tomato___Target_Spot": """
                        Diagnosis: Target spot is caused by *Corynespora cassiicola*.
                        \nAmharic (ምርመራ): ታርጌት ስፖት በ Corynespora cassiicola የሚመጣ ነው።
                        \nCause: Warm, moist conditions and poor airflow.
                        \nAmharic (መንስኤ): ሞቃታማ እና እርጥብ ሁኔታ እንዲሁም የአየር ዝውውር ማጣት።
                        \nTreatment: Apply appropriate fungicides like chlorothalonil.
                        \nAmharic (ህክምና): እንደ ክሎሮታሎኒል ያሉ መድሃኒቶችን መጠቀም።
                        \nPrevention: Increase plant spacing, ensure good drainage.
                        \nAmharic (መከላከያ): ርቀትን መጨመር እና የውሃ ፍሳሽን ማስተካከል።
                        """,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": """
                        Diagnosis: Tomato Yellow Leaf Curl Virus (TYLCV) causes leaf curling and yellowing.
                        \nAmharic (ምርመራ): ቢጫ የቅጠል መጠቅለል ቫይረስ ቅጠልን ያጠቅልላል እና ያቢጫል።
                        \nCause: Spread by whiteflies, especially in hot and dry climates.
                        \nAmharic (መንስኤ): በነጭ ዝንቦች አማካኝነት ይተላለፋል።
                        \nTreatment: No cure — infected plants should be removed immediately.
                        \nAmharic (ህክምና): ፈውስ የለውም፤ የታመሙትን ወዲያውኑ መንቀል።
                        \nPrevention: Use whitefly-resistant tomato varieties.
                        \nAmharic (መከላከያ): በሽታውን የሚቋቋሙ ዝርያዎችን መጠቀም።
                        """,
    "Tomato___Tomato_mosaic_virus": """
                        Diagnosis: Tomato Mosaic Virus leads to mottled or curled leaves.
                        \nAmharic (ምርመራ): ሞዛይክ ቫይረስ ቅጠሎች እንዲቆረቆሩ ያደርጋል።
                        \nCause: Spread by contaminated tools, hands, or seeds.
                        \nAmharic (መንስኤ): በተበከሉ መሳሪያዎች፣ በእጅ ንክኪ ወይም በዘር ይተላለፋል።
                        \nTreatment: No chemical cure — remove infected plants.
                        \nAmharic (ህክምና): ፈውስ የለውም፤ የታመሙትን ማስወገድ።
                        \nPrevention: Wash hands before handling, sterilize equipment.
                        \nAmharic (መከላከያ): እጅን መታጠብ እና መሳሪያዎችን ማጽዳት።
                        """,
    "Tomato___healthy": "✅ The tomato plant is healthy. (ቲማቲሙ ጤናማ ነው።) Keep monitoring for early signs."
}


with tab1:
    @st.cache_resource
    def load_my_model1():
        model = load_model("models/plant disease detection model(CNN MobileNetV2 model).keras", compile=False)
        return model


    model = load_my_model1()
    st.success(f"🌱 You are now using the CNN MobileNetV2 model!!. / አሁን የ CNN MobileNetV2  ሞዴልን እየተጠቀሙ ነው። ")
    st.write("Upload a leaf image and detect the plant disease. / የቅጠል ምስል ይጫኑ እና የተክል በሽታውን ይወቁ።")
    st.write("TIP: For best results, use a clean background (black or white). / ምክር፡ ለተሻለ ውጤት ንጹሕ ጀርባ ይጠቀሙ (ጥቁር ወይም ነጭ )።")
    
    # -----------------------------
    # Image preprocessing
    # -----------------------------
    def preprocess_image(img: Image.Image):
        img = img.convert("RGB")
        img = img.resize((128, 128))
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)  # shape: (1, H, W, 3)
        return x


    # -----------------------------
    # Streamlit UI
    # -----------------------------

    uploaded_file1 = st.file_uploader("Upload plant image (የተክል ምስል ይጫኑ):", type=["jpg", "jpeg", "png"])

    if uploaded_file1 and model:
        try:
            # Load image
            img = Image.open(uploaded_file1).convert("RGB")
            st.image(img, caption="Uploaded Image (የተጫነ ምስል)", use_container_width=True)

            # Preprocess
            x = preprocess_image(img)

            # Predict
            with st.spinner("Analyzing image (ምስሉን በመተንተን ላይ)..."):
                preds = model.predict(x)
                pred_idx = np.argmax(preds)
                pred_class = class_names[pred_idx]
                confidence = np.max(preds)

            # Display results
            st.success(f"Prediction (ግምት ውጤት): {pred_class}")
            st.write(f"Confidence (የመተማመን መጠን): {confidence:.2f}")

            # Show disease info
            with st.expander("💬 Disease Info (የበሽታ መረጃ)"):
                response = disease_responses.get(pred_class, "No additional info available(ተጨማሪ መረጃ አልተገኘም).")
                st.markdown(response)
        except Exception as e:
            st.error(f"Prediction error (የግምት ስህተት): {e}")

with tab2:
    @st.cache_resource
    def load_my_model2():
        interpreter = tf.lite.Interpreter(model_path="models/plant disease detection model(CNN) model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details


    interpreter, input_details, output_details = load_my_model2()

    st.success(f"🌱 You are now using the CNN model!! / አሁን የ CNN ሞዴልን እየተጠቀሙ ነው።")
    st.write("Upload a leaf image and detect the plant disease. / የቅጠል ምስል ይጫኑ እና የተክል በሽታውን ይወቁ።")
    st.write("TIP: For best results, use a clean background (black or white). / ምክር፡ ለተሻለ ውጤት ንጹሕ ጀርባ ይጠቀሙ (ጥቁር ወይም ነጭ )።")
    

    # -----------------------------
    # Image preprocessing
    # -----------------------------
    def preprocess_image(img: Image.Image):
        img = img.convert("RGB")
        img = img.resize((128, 128))
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)  # shape: (1, H, W, 3)
        return x


    # -----------------------------
    # Streamlit UI
    # -----------------------------

    uploaded_file2= st.file_uploader("Upload an image of your plant's leaf (የተክልዎን ቅጠል ምስል ይጫኑ) :", type=["jpg", "jpeg", "png"])

    if uploaded_file2 and model:
        try:
            # Load image
            img = Image.open(uploaded_file2).convert("RGB")
            st.image(img, caption="Uploaded Image (የተጫነ ምስል)", use_container_width=True)

            # Preprocess
            x = preprocess_image(img)
            interpreter.set_tensor(input_details[0]["index"], x)
            interpreter.invoke()

            # Predict
            with st.spinner("Analyzing image (ምስሉን በመተንተን ላይ)..."):
                preds = interpreter.get_tensor(output_details[0]["index"])
                pred_idx = np.argmax(preds)
                pred_class = class_names[pred_idx]
                confidence = np.max(preds)

            # Display results
            st.success(f"Prediction (ግምት ውጤት): {pred_class}")
            st.write(f"Confidence (የመተማመን መጠን): {confidence:.2f}")

            # Show disease info
            with st.expander("💬 Disease Info (የበሽታ መረጃ)"):
                response = disease_responses.get(pred_class, "No additional info available.")
                st.markdown(response)

        except Exception as e:
            st.error(f"Prediction error (የግምት ስህተት): {e}")
