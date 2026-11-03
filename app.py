import streamlit as st
import PyPDF2
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="AI Smart Resume & ATS Recruiter System",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling and Animations
st.markdown("""
<style>
    .main-title {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem !important;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f1f5f9);
        border: 1px solid #cbd5e1;
        padding: 18px 12px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.16);
    }
    .metric-num {
        font-size: 2rem;
        font-weight: 800;
        color: #1e293b;
    }
    .company-box {
        background: #f8fafc;
        border-left: 5px solid #06b6d4;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .badge {
        display: inline-block;
        padding: 5px 12px;
        margin: 4px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        transition: transform 0.15s ease-in-out;
    }
    .badge:hover {
        transform: scale(1.05);
    }
    .badge-found {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .badge-missing {
        background: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }
    .badge-req {
        background: #e0e7ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
    }
</style>
""", unsafe_allow_html=True)

# Global Shared Storage across sessions using Streamlit Cache
@st.cache_resource
def get_global_company_database():
    return {
        "Google | Python Backend Engineer": {
            "company": "Google",
            "role": "Python Backend Engineer",
            "description": "Looking for a backend engineer experienced in Python, Django, FastAPI, PostgreSQL, Docker, Git, REST APIs, and Linux architecture.",
            "skills": ["python", "django", "fastapi", "sql", "docker", "git", "rest api", "linux"],
            "min_score": 75,
            "roadmap": "Focus on high-concurrency microservices, async APIs with FastAPI, and Docker deployments."
        },
        "Amazon | Python Data Analyst": {
            "company": "Amazon",
            "role": "Python Data Analyst",
            "description": "Seeking analyst skilled in Python, SQL, Pandas, NumPy, Power BI, Statistics, and Data Wrangling.",
            "skills": ["python", "sql", "pandas", "numpy", "power bi", "statistics", "excel"],
            "min_score": 70,
            "roadmap": "Strengthen complex SQL joins/window functions and build executive dashboards using Power BI."
        },
        "TCS | Full Stack Web Developer": {
            "company": "TCS",
            "role": "Full Stack Web Developer",
            "description": "Requires expertise in React, Node.js, Express, MongoDB, JavaScript, HTML, CSS, Git, and RESTful APIs.",
            "skills": ["javascript", "react", "node.js", "mongodb", "express", "html", "css", "git"],
            "min_score": 65,
            "roadmap": "Master React hooks/state management and secure JWT authentication backend flows."
        }
    }

GLOBAL_COMPANIES = get_global_company_database()

# PDF Text Extraction with Scanned/Empty Guardrail
def extract_pdf_text(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        extracted_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + " "
        return extracted_text.strip()
    except Exception:
        return ""

def clean_text(raw_text):
    text = raw_text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_similarity(resume_text, benchmark_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, benchmark_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(score) * 100, 2)

def analyze_skills(cleaned_text, target_skills):
    detected = []
    missing = []
    for skill in target_skills:
        pattern = r"(?i)\b" + re.escape(skill) + r"\b"
        if re.search(pattern, cleaned_text):
            detected.append(skill.upper())
        else:
            missing.append(skill.upper())
    match_pct = round((len(detected) / len(target_skills)) * 100, 1) if target_skills else 0.0
    return detected, missing, match_pct

# Sidebar Navigation
st.sidebar.markdown("## 🧭 Navigation Portal")
user_role = st.sidebar.radio(
    "Choose Active Workspace:",
    ["👨‍🎓 Candidate Mode (Screen Resume)", "🏢 Recruiter / Company Mode (Post Requirements)"]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"**Total Active Job Openings:** `{len(GLOBAL_COMPANIES)}`")
st.sidebar.caption("System: Scikit-learn TF-IDF & Vector Similarity")

# Title Header
st.markdown('<p class="main-title">🎯 AI-Driven Smart Resume & Recruiter ATS Platform</p>', unsafe_allow_html=True)
st.caption("Multi-Company Requirement Portal & Candidate Gap Analysis Engine")
st.write("")

# ----------------- 🏢 RECRUITER / COMPANY MODE -----------------
if user_role == "🏢 Recruiter / Company Mode (Post Requirements)":
    st.subheader("🏢 Company Requirement Management Portal")
    st.info("💡 HR aur Technical Recruiters yahan nayi vacancies, mandatory tech stack aur minimum cutoff define kar sakte hain.")

    with st.form("company_posting_form", clear_on_submit=True):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            comp_name = st.text_input("🏢 Company Name:", placeholder="e.g., Infosys, Microsoft, Wipro")
            job_title = st.text_input("💼 Job Title / Role:", placeholder="e.g., Python Backend Engineer")
        with col_c2:
            min_cutoff = st.slider("🎯 Minimum ATS Cutoff Score (%):", min_value=40, max_value=90, value=70)
            skills_input = st.text_input(
                "🔑 Mandatory Tech Stack (comma-separated):",
                placeholder="e.g., python, django, rest api, docker, sql"
            )

        job_desc = st.text_area(
            "📝 Detailed Job Description / Scope of Work:",
            placeholder="Paste complete job overview, expected responsibilities, and qualifications..."
        )
        custom_roadmap = st.text_input(
            "💡 Preparation Tip / Learning Roadmap for Candidates:",
            placeholder="e.g., Focus on microservices architecture, API security, and database indexing."
        )

        submit_btn = st.form_submit_button("🚀 Publish & Save Company Requirement")

        if submit_btn:
            if not comp_name.strip() or not job_title.strip() or not skills_input.strip() or not job_desc.strip():
                st.error("⚠️ Please fill in all required fields (Company, Title, Skills, Description).")
            else:
                profile_key = f"{comp_name.strip()} | {job_title.strip()}"
                skill_list = [s.strip().lower() for s in skills_input.split(",") if s.strip()]
                
                # Save into Global Shared Cache
                GLOBAL_COMPANIES[profile_key] = {
                    "company": comp_name.strip(),
                    "role": job_title.strip(),
                    "description": job_desc.strip(),
                    "skills": skill_list,
                    "min_score": min_cutoff,
                    "roadmap": custom_roadmap.strip() if custom_roadmap.strip() else "Brush up on core competencies required for this role."
                }
                st.success(f"🎉 Successfully published **{profile_key}**! Ye opening ab sabhi candidates ke liye live hai.")

    st.write("---")
    st.subheader("📋 Currently Active Company Openings")
    comp_list = []
    for k, v in GLOBAL_COMPANIES.items():
        comp_list.append({"Company": v["company"], "Role": v["role"], "Required Skills Count": len(v["skills"]), "Cutoff": f"{v['min_score']}%"})
    st.dataframe(pd.DataFrame(comp_list), use_container_width=True)

# ----------------- 👨‍🎓 CANDIDATE MODE -----------------
else:
    st.subheader("🎯 Candidate Evaluation & Gap Analysis")
    
    available_jobs = list(GLOBAL_COMPANIES.keys())
    selected_job_key = st.selectbox("Select Target Company & Opening to Apply For:", available_jobs)
    job_data = GLOBAL_COMPANIES[selected_job_key]

    # Show company requirements card
    st.markdown('<div class="company-box">', unsafe_allow_html=True)
    st.markdown(f"### 🏢 {job_data['company']} — Requirement Overview")
    st.write(f"**Target Role:** `{job_data['role']}` | **Required ATS Cutoff:** `{job_data['min_score']}%`")
    st.caption(f"**Company JD:** {job_data['description']}")
    
    badges_req = "".join([f'<span class="badge badge-req">{s.upper()}</span>' for s in job_data["skills"]])
    st.markdown(f"**Mandatory Competencies:**<br>{badges_req}", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Resume Upload
    col_up, col_demo = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader("📂 Upload Your Resume (PDF Format)", type=["pdf"])
    with col_demo:
        st.write("**Quick Test**")
        use_sample = st.button("⚡ Test with Sample Resume")

    resume_raw_text = ""
    is_scanned_pdf = False

    if uploaded_file is not None:
        extracted = extract_pdf_text(uploaded_file)
        if not extracted or len(extracted) < 15:
            is_scanned_pdf = True
        else:
            resume_raw_text = extracted
    elif use_sample:
        resume_raw_text = """
        Ayush Dwivedi - Software Trainee
        Technical Skills: Python, SQL, Git, basic Django, Data structures, Regex, REST API.
        Projects: Student Management System in Python using SQLite, Portfolio Website.
        Education: Bachelor of Computer Applications (BCA).
        """

    # Scanned PDF Warning Banner (Guardrail against prank files)
    if is_scanned_pdf:
        st.warning("⚠️ **Notice: No Machine-Readable Text Found in PDF!**")
        st.info("Aapka PDF file ek image/scanned screenshot lag raha hai. Abhi ke prototype model ke liye please selectable text PDF upload karein. *(Scanned image OCR parsing Major Project 6th Sem mein scheduled hai).*")

    # Scoring & Gap Analysis
    elif resume_raw_text:
        with st.spinner("Analyzing semantics, matching company keywords, and calculating ATS fit..."):
            cleaned = clean_text(resume_raw_text)
            semantic_score = compute_similarity(cleaned, clean_text(job_data["description"]))
            detected_skills, missing_skills, skill_score = analyze_skills(cleaned, job_data["skills"])
            overall_ats_score = round((skill_score * 0.7) + (semantic_score * 0.3), 1)

        st.success("✅ Resume Successfully Screened!")

        # Dynamic KPI Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-num">{overall_ats_score}%</div>
                <div style="font-weight:700; color:#475569;">Overall ATS Score</div>
                <small style="color:#64748b;">Hybrid Match</small>
            </div>
            ''', unsafe_allow_html=True)

        with m2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-num">{skill_score}%</div>
                <div style="font-weight:700; color:#475569;">Skill Coverage</div>
                <small style="color:#64748b;">{len(detected_skills)} of {len(job_data['skills'])} Found</small>
            </div>
            ''', unsafe_allow_html=True)

        with m3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-num">{semantic_score}%</div>
                <div style="font-weight:700; color:#475569;">Context Relevance</div>
                <small style="color:#64748b;">TF-IDF Vector Score</small>
            </div>
            ''', unsafe_allow_html=True)

        with m4:
            cutoff_diff = round(overall_ats_score - job_data["min_score"], 1)
            cutoff_color = "#15803d" if cutoff_diff >= 0 else "#b91c1c"
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-num" style="color: {cutoff_color};">{job_data['min_score']}%</div>
                <div style="font-weight:700; color:#475569;">Company Cutoff</div>
                <small style="color:#64748b;">{'Passed Criteria' if cutoff_diff >= 0 else 'Below Cutoff'}</small>
            </div>
            ''', unsafe_allow_html=True)

        st.write("")
        st.progress(min(int(overall_ats_score), 100))

        # Shortlist Decision Banner
        if overall_ats_score >= job_data["min_score"]:
            st.success(f"🌟 **Shortlisted for {job_data['company']}!** Your profile cleared the minimum ATS threshold of {job_data['min_score']}%.")
        else:
            st.error(f"🚨 **Not Shortlisted:** Candidate scored {overall_ats_score}%, which is below {job_data['company']}'s cutoff of {job_data['min_score']}%.")

        st.write("---")

        # Skills Breakdown Cards
        col_det, col_mis = st.columns(2)
        with col_det:
            st.markdown(f"#### ✅ Matched Requirements ({len(detected_skills)})")
            if detected_skills:
                badges = "".join([f'<span class="badge badge-found">{s}</span>' for s in detected_skills])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.info("No mandatory skills found in resume.")

        with col_mis:
            st.markdown(f"#### ❌ Missing Requirements ({len(missing_skills)})")
            if missing_skills:
                badges = "".join([f'<span class="badge badge-missing">{s}</span>' for s in missing_skills])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.success("Candidate matches 100% of required competencies!")

        st.write("---")

        # Upskilling Roadmap
        st.markdown(f"#### 💡 Preparation Roadmap for {job_data['company']}")
        st.info(job_data["roadmap"])

        # Download Report
        report_data = (
            f"=== {job_data['company'].upper()} ATS SCREENING REPORT ===\n"
            f"Target Role: {job_data['role']}\n"
            f"Candidate Score: {overall_ats_score}% (Cutoff: {job_data['min_score']}%)\n"
            f"Shortlist Status: {'SHORTLISTED' if overall_ats_score >= job_data['min_score'] else 'REJECTED'}\n\n"
            f"Matched Skills: {', '.join(detected_skills)}\n"
            f"Missing Skills: {', '.join(missing_skills)}\n\n"
            f"Recommended Roadmap: {job_data['roadmap']}\n"
        )
        st.download_button(
            "📥 Download Candidate Screening Report (.txt)",
            report_data,
            file_name=f"Report_{job_data['company']}.txt",
            mime="text/plain"
        )
