import streamlit as st
import PyPDF2
import docx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Code by Diego Gonzalez Farias, Personal Project 12/03/2025

# Function to extract text from uploaded file
def extract_text_from_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"❌ Error extracting text from file: {e}")
    return text


# Function to analyze ATS compatibility
def analyze_ats_compatibility(job_description, cv_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([job_description, cv_text])

    # Calculate similarity score
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

    # Extract keywords
    feature_names = vectorizer.get_feature_names_out()
    job_tfidf = tfidf_matrix.toarray()[0]
    cv_tfidf = tfidf_matrix.toarray()[1]

   #Dataframe using keywords and TFIDF model
    keywords_df = pd.DataFrame({'Keyword': feature_names, 'Job_TFIDF': job_tfidf, 'CV_TFIDF': cv_tfidf})
    keywords_df['Difference'] = abs(keywords_df['Job_TFIDF'] - keywords_df['CV_TFIDF'])
    keywords_df = keywords_df.sort_values(by='Difference', ascending=False).head(20)  # Top missing keywords

    return similarity, keywords_df



def generate_feedback(similarity, missing_keywords):
    st.subheader("📝 ATS Optimization Tips")

    if similarity < 50:
        st.error("❌ **Low ATS Match** – Your CV might be rejected by the ATS.")
        st.write("""
        🔹 **Your CV is missing many key terms from the job description.**  
        🔹 **Recommendations:**  
        - Add **relevant skills and job-specific keywords**.
        - Ensure **work experience descriptions** include **measurable achievements**.
        - Check for **formatting issues** (avoid tables, columns, or graphics).
        """)
    elif 50 <= similarity < 80:
        st.warning("⚠️ **Medium ATS Match** – Some improvements needed.")
        st.write("""
        🔹 **Your CV includes relevant terms but is missing some critical keywords.**  
        🔹 **Recommendations:**  
        - Add more **technical skills & industry-specific terms**.
        - Ensure **job titles** match what is in the job description.
        - Use **standard section titles** (Work Experience, Education, Skills).
        """)
    else:
        st.success("✅ **High ATS Match** – Your CV is well-optimized!")
        st.write("""
        🎯 **Your CV is well-optimized for this job!**  
        🔹 **Final Recommendations:**  
        - Make sure **important skills appear multiple times** in your CV.  
        - Keep formatting clean & ATS-friendly.  
        - Ensure your **top skills** appear in the **first half** of your CV.  
        """)

    if not missing_keywords.empty:
        st.subheader("🔑 Top Missing Keywords")
        st.write("Consider adding these terms to improve your ATS ranking:")
        st.dataframe(missing_keywords)


#Heatmap function plot
def plot_heatmap(keywords_df):
    plt.figure(figsize=(8, 5))
    sns.heatmap(keywords_df.set_index("Keyword"), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("ATS Keyword Match Heatmap")
    st.pyplot(plt)


# Streamlit section
st.title("📄 ATS CV Analyzer: Will Your Resume Pass the Screening?")
st.subheader("By: Diego Gonzalez Farias")

st.markdown("""
## 🎯 What This Tool Does
This ATS Analyzer checks if your CV has the **right keywords** to pass through **Applicant Tracking Systems (ATS)**. 

🔹 **Upload your CV** (PDF, DOCX, or TXT).  
🔹 **Enter a Job Description** you’re applying for.  
🔹 **Get a similarity score** showing how well your CV matches the job.  
🔹 **See missing keywords** to improve your CV.  
🔹 **Optimize your resume** for better ATS ranking!

📌 **Tip:** The higher the match, the better your chances of getting past an ATS!  
""")

#Input box to enter job description
st.subheader("🔹 Enter the Job Description:")
job_description = st.text_area("Paste the job description here...")


st.subheader("🔹 Upload Your CV (PDF, DOCX, or TXT):")
uploaded_file = st.file_uploader("Upload CV", type=["pdf", "docx", "txt"])

if uploaded_file is not None and job_description.strip():
    with st.spinner("Analyzing CV..."):
        cv_text = extract_text_from_file(uploaded_file)


        st.subheader("📄 Extracted CV Text Preview (What an ATS Sees)")
        st.text_area("This is the text your ATS will read:", cv_text[:2000], height=300)

        # Calculate ATS similarity
        similarity_percentage, keywords_df = analyze_ats_compatibility(job_description, cv_text)

  #ATS SCORE bsed on similarity percentage
        st.success(f"✅ Your CV matches **{similarity_percentage:.2f}%** with the job description.")

        # Provide ATS Optimization Feedback
        generate_feedback(similarity_percentage, keywords_df)

        # Show heatmap of keyword similarities
        st.subheader("📊 ATS Keyword Match Heatmap")
        plot_heatmap(keywords_df)

        # Download button for missing keywords
        csv = keywords_df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Missing Keywords CSV", csv, "Missing_Keywords.csv", "text/csv")

st.markdown("🔗 Connect with me on [LinkedIn](https://www.linkedin.com/in/diego-gonzalez-farias-248870234/)")

