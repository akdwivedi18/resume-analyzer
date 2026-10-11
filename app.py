import streamlit as st
import PyPDF2
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="AI Resume & Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
    }
    .skill-badge-found {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
    }
    .skill-badge-missing {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Domain Benchmarks Database
JOB_BENCHMARKS = {
    "Python Developer": {
        "description": "Python developer experienced in Django, Flask, REST APIs, SQL, Git, Docker, OOP concepts, Linux and unit testing.",
        "skills": ["python", "django", "flask", "sql", "git", "rest api", "docker", "oops", "linux"],
        "roadmap": "Focus on backend architecture, building RESTful APIs, database ORM queries, and Docker containerization."
    },
    "Data Analyst": {
        "description": "Data analyst skilled in Python, SQL, Excel, Pandas, NumPy, Power BI, Tableau, data visualization and statistics.",
        "skills": ["python", "sql", "excel", "pandas", "numpy", "power bi", "tableau", "statistics", "data visualization"],
        "roadmap": "Strengthen SQL complex queries, Pandas data wrangling, and interactive dashboards in Power BI."
    },
    "Web Developer (Full Stack)": {
        "description": "Full stack web developer with expertise in HTML, CSS, JavaScript, React, Node.js, Express, MongoDB and Git.",
        "skills": ["html", "css", "javascript", "react", "node.js", "mongodb", "git", "bootstrap", "tailwind"],
        "roadmap": "Deepen React state management, master backend CRUD APIs with Node/Express, and build full stack projects."
    },
    "Java Developer": {
        "description": "Java developer proficient in Core Java, Spring Boot, Hibernate, MySQL, OOP principles, Maven and REST APIs.",
        "skills": ["java", "spring boot", "hibernate", "sql", "oops", "microservices", "git", "maven", "rest api"],
        "roadmap": "Master Core Java collections, Spring Boot JPA repositories, and building microservices."
    }
}

# Function: PDF Text Extraction
def extract_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text += page_text + " "
    return extracted_text

# Function: Clean Text
def clean_text(raw_text):
    text = raw_text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function: Cosine Similarity Scoring
def compute_similarity(resume_text, benchmark_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, benchmark_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(score) * 100, 2)

# Function: Skill Gap Identification
def analyze_skills(cleaned_text, target_skills):
    detected = []
    missing = []
    for skill in target_skills:
        pattern = r"(?i)\b" + re.escape(skill) + r"\b"
        if re.search(pattern, cleaned_text):
            detected.append(skill.upper())
        else:
            missing.append(skill.upper())
    
    match_pct = round((len(detected) / len(target_skills)) * 100, 1)
    return detected, missing, match_pct

# UI Header
st.title("🎯 AI-Driven Smart Resume & Skill Gap Analyzer")
st.markdown("**Automated ATS Screening and Competency Assessment Pipeline**")
st.write("---")

# Sidebar Configuration
st.sidebar.header("📋 Role Configuration")
selected_role = st.sidebar.selectbox("Target Job Profile:", list(JOB_BENCHMARKS.keys()))
role_data = JOB_BENCHMARKS[selected_role]

st.sidebar.markdown(f"**Benchmark Skills:** `{len(role_data['skills'])}` Core Requirements")
st.sidebar.markdown("---")
st.sidebar.caption("• PyPDF2 Parser\n• TF-IDF Vectorizer\n• Cosine Similarity Engine")

# Main Screen
col_up, col_demo = st.columns([3, 1])
with col_up:
    uploaded_file = st.file_uploader("Upload Applicant Resume (PDF Format)", type=["pdf"])
with col_demo:
    st.write("**Quick Test**")
    use_sample = st.button("Use Sample Resume")

resume_raw_text = ""
if uploaded_file is not None:
    resume_raw_text = extract_pdf_text(uploaded_file)
elif use_sample:
    resume_raw_text = """
    Ayush Dwivedi - BCA Final Year Student
    Skills: Python programming, SQL, HTML, CSS, Git, basic Django, Data structures, C language.
    Projects: Student Management System in Python with SQLite, Portfolio Website.
    Education: Bachelor of Computer Applications (BCA).
    """

if resume_raw_text:
    cleaned = clean_text(resume_raw_text)
    semantic_score = compute_similarity(cleaned, clean_text(role_data["description"]))
    detected_skills, missing_skills, skill_score = analyze_skills(cleaned, role_data["skills"])
    overall_ats_score = round((skill_score * 0.7) + (semantic_score * 0.3), 1)

    st.success("✅ Document Parsed and Evaluated Successfully!")

    # KPI Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Overall ATS Score", f"{overall_ats_score}%")
        st.caption("Weighted Hybrid Metric")
        st.markdown('</div>', unsafe_allow_html=True)

    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Core Skill Coverage", f"{skill_score}%")
        st.caption(f"{len(detected_skills)} of {len(role_data['skills'])} Skills Found")
        st.markdown('</div>', unsafe_allow_html=True)

    with m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Contextual Relevance", f"{semantic_score}%")
        st.caption("TF-IDF Vector Cosine Alignment")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    if overall_ats_score >= 70:
        st.success("🌟 **Strong ATS Fit:** High keyword alignment with target profile.")
    elif overall_ats_score >= 40:
        st.warning("⚠️ **Moderate Fit:** Lacks key industry tools.")
    else:
        st.error("🚨 **Low Alignment:** Significant skill gaps identified.")

    st.write("---")

    # Skills Display
    c_detected, c_missing = st.columns(2)
    with c_detected:
        st.subheader(f"✅ Identified Competencies ({len(detected_skills)})")
        if detected_skills:
            badges_html = "".join([f'<span class="skill-badge-found">{s}</span>' for s in detected_skills])
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.write("No matching domain skills identified.")

    with c_missing:
        st.subheader(f"❌ Missing Benchmark Skills ({len(missing_skills)})")
        if missing_skills:
            badges_html = "".join([f'<span class="skill-badge-missing">{s}</span>' for s in missing_skills])
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.write("All required benchmark skills are present!")

    st.write("---")

    # Roadmap & Download
    st.subheader("💡 Strategic Upskilling Roadmap")
    st.info(role_data["roadmap"])

    report_text = f"ATS REPORT\\nRole: {selected_role}\\nScore: {overall_ats_score}%\\nDetected: {', '.join(detected_skills)}\\nMissing: {', '.join(missing_skills)}"
    st.download_button("📥 Download Analysis Report (.txt)", report_text, file_name=f"Report_{selected_role}.txt")
