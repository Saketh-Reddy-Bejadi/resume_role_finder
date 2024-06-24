import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    """
    Function to clean resume text by removing URLs, special characters, etc.
    """
    clean_text = re.sub('http\S+\s*', ' ', resume_text)  # Remove URLs
    clean_text = re.sub('RT|cc', ' ', clean_text)  # Remove RT and cc
    clean_text = re.sub('#\S+', '', clean_text)  # Remove hashtags
    clean_text = re.sub('@\S+', '  ', clean_text)  # Remove mentions
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)  # Remove special characters
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)  # Remove non-ASCII characters
    clean_text = re.sub('\s+', ' ', clean_text)  # Replace multiple spaces with a single space
    return clean_text

def extract_text_from_pdf(uploaded_file):
    """
    Function to extract text from a PDF file
    """
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_skills_and_experience(text):
    """
    Function to extract skills and experience from resume text
    """
    skills_keywords = ['Python', 'Java', 'C++', 'SQL', 'JavaScript', 'AWS', 'Azure', 'Docker', 'Kubernetes', 'Machine Learning', 'Deep Learning', 'AI', 'Data Science', 'DevOps', 'Web Development', 'React', 'Angular', 'Django', 'Flask']
    experience_keywords = ['experience', 'worked', 'developed', 'managed', 'led', 'designed']

    skills = [skill for skill in skills_keywords if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE)]
    experience = [exp for exp in experience_keywords if re.search(r'\b' + re.escape(exp) + r'\b', text, re.IGNORECASE)]

    return skills, experience

def main():
    """
    Main function to run the Streamlit web app
    """
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_bytes = uploaded_file.read()
            try:
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        skills, experience = extract_skills_and_experience(cleaned_resume)
        
        st.write("Extracted Skills:", skills)
        st.write("Extracted Experience Keywords:", experience)
        
        # Combine skills and experience into the cleaned_resume
        enriched_resume = " ".join(skills + experience) + " " + cleaned_resume
        input_features = tfidf.transform([enriched_resume])
        prediction_id = clf.predict(input_features)[0]

        # Updated category mapping with technical roles only
        category_mapping = {
            2: "Automation Testing",
            3: "Blockchain",
            4: "Business Analyst",
            5: "Civil Engineer",
            6: "Data Science",
            7: "Database",
            8: "DevOps Engineer",
            9: "DotNet Developer",
            10: "ETL Developer",
            11: "Electrical Engineer",
            13: "Hadoop",
            15: "Java Developer",
            16: "Mechanical Engineer",
            17: "Network Security Engineer",
            18: "Operations Manager",
            19: "PMO",
            20: "Python Developer",
            21: "SAP Developer",
            23: "Testing",
            24: "Web Designing",
            25: "Web Developer",
            26: "AI Engineer",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)

if __name__ == "__main__":
    main()
