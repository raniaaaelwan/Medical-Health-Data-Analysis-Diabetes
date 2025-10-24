import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google import genai
import numpy as np

# ---------------------------
# 1. Gemini Client
# ---------------------------
client = genai.Client(api_key="AIzaSyBNuE1WsAeEKbYne0apxuZnVUCvM2bW7BY")  

# ---------------------------
# 2. Load dataset
# ---------------------------
@st.cache_data
def load_data():
    path = r"C:\Users\Asus\Documents\Diabetes\diabetes_binary_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(path)
    df['Sex_label'] = df['Sex'].map({0: 'Female', 1: 'Male'})
    age_labels_en = {
        1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
        6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
        11: "70–74", 12: "75–79", 13: "80+"
    }
    df['Age_label'] = df['Age'].map(age_labels_en)
    return df

df = load_data()

# ---------------------------
# 3. Streamlit setup
# ---------------------------
st.set_page_config(page_title="Diabetes & Health Dashboard", page_icon="💉", layout="wide")

# ---------------------------
# 4. Custom CSS
# ---------------------------
st.markdown("""
<style>
.banner { background: linear-gradient(90deg, #ffafcc, #a2d2ff); color:white; padding:2rem 1rem;
border-radius:0 0 25px 25px; text-align:center; box-shadow:0px 3px 10px rgba(0,0,0,0.1); }
.banner h1 { font-size:2.3rem; margin-bottom:0.4rem; font-weight:800; }
.banner p { font-size:1.1rem; opacity:0.95; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #fce4ec 0%, #e3f2fd 100%); }
div.stButton > button { background: linear-gradient(90deg, #ffafcc, #a2d2ff); color:white; border-radius:10px;
height:3em; width:100%; font-weight:600; border:none; transition:all 0.25s ease-in-out; box-shadow:0px 2px 6px rgba(0,0,0,0.1);}
div.stButton > button:hover { transform:scale(1.03); background: linear-gradient(90deg, #f48fb1, #90caf9); }
.response-box { background-color:#f8faff; border-radius:12px; border-left:6px solid #a2d2ff;
padding:1.2rem; margin-top:1rem; font-size:1.05rem; box-shadow:0px 2px 8px rgba(255,182,193,0.2); }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 5. Banner
# ---------------------------
st.markdown("""
<div class="banner">
    <h1>🩺 Diabetes Dataset Explorer</h1>
    <p>Ask Questions About Diabetes Dataset</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# 6. Sidebar Filters
# ---------------------------
st.sidebar.header("🔧 Filters")

# Age filter
age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
selected_age = st.sidebar.slider("Select Age Range:", age_min, age_max, (age_min, age_max))

# Gender filter
gender_options = df['Sex_label'].unique().tolist()
selected_gender = st.sidebar.multiselect("Select Gender:", options=gender_options, default=gender_options)

# Smoker filter
smoker_options = df['Smoker'].unique().tolist()
selected_smoker = st.sidebar.multiselect("Select Smoker Status:", options=smoker_options, default=smoker_options)

# Apply filters
filtered_df = df[
    (df['Age'] >= selected_age[0]) & (df['Age'] <= selected_age[1]) &
    (df['Sex_label'].isin(selected_gender)) &
    (df['Smoker'].isin(selected_smoker))
]

# ---------------------------
# 7. Sidebar Preset Questions
# ---------------------------
st.sidebar.header("💡 Ask a Question")
preset_questions = [
    "How does fruit consumption affect mental health?",
    "Is there a link between income and insurance coverage?",
    "Do people with higher education have better physical health?",
    "What factors are most associated with difficulty walking?",
    "How does BMI relate to diabetes likelihood?",
    "Are smokers at higher risk of diabetes complications?",
    "What lifestyle factors most influence diabetes outcomes?"
]
selected_question = st.sidebar.radio("Choose a question:", options=[""] + preset_questions)

if selected_question:
    user_question = selected_question
else:
    user_question = st.text_area("💬 Type your own question below:")
# ai chatbot answer
sample_data = filtered_df.head(20).to_csv(index=False)
if st.button("🔍 Get Answer"):
    if not user_question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("🔬 Analyzing data ..."):
            prompt = f"""
            You are a medical data analyst specializing in diabetes research.
            Here is a small sample of the dataset:

            {sample_data}

            Question: {user_question}

            Please answer clearly, in a professional and human-friendly way suitable for healthcare insights.
            """

            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                st.markdown('<div class="response-box">', unsafe_allow_html=True)
                st.markdown("### ✅ Analysis:")
                st.write(response.text)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------------------
# 8. Dashboard Charts
st.markdown("""
<div class="banner">
    <h1>📊 Interactive Diabetes Dashboard</h1>
    <p>Filter data, visualize charts, and ask AI-powered questions</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"### Showing {filtered_df.shape[0]} records after filtering")

# ---------------------------
soft_palette = ['#6CA6CD', '#FFB6C1']

# Use columns for dashboard layout
col1, col2 = st.columns(2)

with col1:
    # Chart 1: Diabetes vs Physical Activity
    st.subheader("🏃 Diabetes vs Physical Activity")
    phys_diab = filtered_df.groupby(['PhysActivity', 'Diabetes_binary']).size().unstack(fill_value=0)
    phys_diab_percent = phys_diab.div(phys_diab.sum(axis=1), axis=0) * 100
    fig1, ax1 = plt.subplots()
    phys_diab_percent.plot(kind='bar', ax=ax1, color=soft_palette)
    ax1.set_xlabel("Physical Activity (0=No, 1=Yes)")
    ax1.set_ylabel("Percentage")
    ax1.legend(title="Diabetes", labels=["No", "Yes"])
    st.pyplot(fig1)

    # Chart 2: Correlation Heatmap
    st.subheader("🧠 Correlation Heatmap")
    selected_cols = ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'PhysActivity',
                     'HvyAlcoholConsump', 'AnyHealthcare', 'GenHlth', 'MentHlth',
                     'PhysHlth', 'DiffWalk']
    corr = filtered_df[selected_cols].corr()
    fig2, ax2 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', annot_kws={"size": 8}, ax=ax2)
    st.pyplot(fig2)

    # Chart 3: Education vs Access to Healthcare
    st.subheader("🎓 Education vs Healthcare")
    edu_health = filtered_df.groupby(['Education', 'AnyHealthcare']).size().unstack(fill_value=0)
    edu_health_percent = edu_health.div(edu_health.sum(axis=1), axis=0) * 100
    fig3, ax3 = plt.subplots()
    edu_health_percent.plot(kind='bar', ax=ax3, color=soft_palette)
    ax3.set_xlabel("Education Level")
    ax3.set_ylabel("Percentage")
    ax3.legend(title="Any Healthcare", labels=["No", "Yes"])
    st.pyplot(fig3)

with col2:
    # Chart 4: Age Group Distribution (Diabetes = Yes)
    st.subheader("🥧 Age Group Distribution (Diabetes=Yes)")
    age_yes = filtered_df[filtered_df['Diabetes_binary']==1]['Age_label'].value_counts().sort_index()
    total = age_yes.sum()
    percentages = age_yes / total * 100
    colors = ['#FFB6C1', '#AEC6CF', '#F4A7B9', '#B0E0E6', '#FFD1DC', '#ADD8E6',
              '#F8BBD0', '#87CEFA', '#FFCCE5', '#B2DFEE', '#FFC0CB', '#AFEEEE', '#F6D1D1']
    def label_filter(pct, allvals): return f"{pct:.1f}%" if pct > 5 else ""
    fig4, ax4 = plt.subplots()
    wedges, texts, autotexts = ax4.pie(
        age_yes,
        labels=age_yes.index,
        autopct=lambda pct: label_filter(pct, age_yes),
        colors=colors,
        startangle=90,
        textprops={'fontsize': 10}
    )
    for i, txt in enumerate(texts):
        if percentages[i] <= 5: txt.set_text("")
    st.pyplot(fig4)

    # Chart 5: Gender Distribution (Diabetes = Yes)
    st.subheader("🥧 Gender Distribution (Diabetes=Yes)")
    gender_yes = filtered_df[filtered_df['Diabetes_binary']==1]['Sex_label'].value_counts()
    fig5, ax5 = plt.subplots()
    ax5.pie(gender_yes, labels=gender_yes.index, autopct='%1.1f%%', colors=soft_palette, startangle=90)
    st.pyplot(fig5)

# Additional charts below dashboard
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    # Chart 6: High Blood Pressure Rate by Age Group (%)
    st.subheader("📈 High Blood Pressure Rate by Age Group")
    bp_rate = filtered_df.groupby('Age_label')['HighBP'].mean().sort_index() * 100
    fig6, ax6 = plt.subplots()
    bp_rate.plot(kind='line', marker='s', color=soft_palette[0], ax=ax6)
    ax6.set_xlabel("Age Group")
    ax6.set_ylabel("HighBP Rate (%)")
    st.pyplot(fig6)

    # Chart 7: Age Group vs High Cholesterol (%)
    st.subheader("📊 Age Group vs High Cholesterol")
    chol_age = filtered_df.groupby(['Age_label', 'HighChol']).size().unstack(fill_value=0)
    chol_age_percent = chol_age.div(chol_age.sum(axis=1), axis=0) * 100
    fig7, ax7 = plt.subplots()
    chol_age_percent.plot(kind='bar', ax=ax7, color=soft_palette)
    ax7.set_xlabel("Age Group")
    ax7.set_ylabel("Percentage")
    ax7.legend(title="High Cholesterol", labels=["No", "Yes"])
    st.pyplot(fig7)

with col4:
    # Chart 8: Average BMI by Age Group
    st.subheader("📈 Average BMI by Age Group")
    avg_bmi = filtered_df.groupby('Age_label')['BMI'].mean().sort_index()
    fig8, ax8 = plt.subplots()
    avg_bmi.plot(kind='line', marker='o', color=soft_palette[1], ax=ax8)
    ax8.set_xlabel("Age Group")
    ax8.set_ylabel("Average BMI")
    st.pyplot(fig8)
