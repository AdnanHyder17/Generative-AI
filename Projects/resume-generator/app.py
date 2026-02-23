import streamlit as st
import asyncio
from src.linkedin_scraper import scrape_linkedin_profile
from src.resume_generator import ResumeGenerator
from src.pdf_exporter import export_resumes_to_pdf
import time

st.set_page_config(
    page_title="AI Resume Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #6B7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .resume-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .resume-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1F2937;
    }
    .badge {
        background: #EEF2FF;
        color: #4F46E5;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar config
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    vllm_url = st.text_input(
        "vLLM Server URL",
        value="http://localhost:8000",
        help="URL of your running vLLM server"
    )
    
    model_name = st.selectbox(
        "Model",
        [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
        ],
        help="Must match the model loaded in vLLM"
    )
    
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens per Resume", 500, 2000, 1200, 100)
    
    st.markdown("---")
    st.markdown("### 📋 Resume Styles")
    st.markdown("""
    Generates 10 tailored versions:
    1. 🎯 ATS-Optimized
    2. 💼 Executive/Leadership
    3. 🚀 Startup/Entrepreneur
    4. 🎨 Creative/Portfolio
    5. 🔬 Technical/Engineering
    6. 📊 Data-Driven/Metrics
    7. 🌍 International/Global
    8. 🏢 Corporate/Traditional
    9. 🔄 Career Change
    10. 📚 Academic/Research
    """)
    
    st.markdown("---")
    st.caption("Built with vLLM + Streamlit")

# Main content
st.markdown('<div class="main-header">🚀 AI Resume Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste your LinkedIn URL → Get 10 tailored resumes in seconds</div>', unsafe_allow_html=True)

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    linkedin_url = st.text_input(
        "LinkedIn Profile URL",
        placeholder="https://www.linkedin.com/in/yourname/",
        label_visibility="collapsed"
    )
with col2:
    generate_btn = st.button("✨ Generate Resumes", type="primary", use_container_width=True)

# Optional manual input
with st.expander("📝 Or manually enter your profile info (no LinkedIn required)"):
    col_a, col_b = st.columns(2)
    with col_a:
        manual_name = st.text_input("Full Name")
        manual_title = st.text_input("Current Title")
        manual_location = st.text_input("Location")
        manual_email = st.text_input("Email")
    with col_b:
        manual_summary = st.text_area("Professional Summary", height=100)
        manual_skills = st.text_area("Skills (comma-separated)", height=60)
    
    manual_experience = st.text_area(
        "Work Experience (paste freely)",
        height=150,
        placeholder="Company | Role | Duration | Achievements..."
    )
    manual_education = st.text_area("Education", height=80)
    use_manual = st.checkbox("Use manual input instead of LinkedIn")

# Generation logic
if generate_btn:
    if not linkedin_url and not use_manual:
        st.warning("⚠️ Please enter a LinkedIn URL or use manual input.")
    else:
        with st.spinner(""):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Step 1: Get profile data
                if use_manual:
                    status_text.text("📥 Collecting profile data...")
                    profile_data = {
                        "name": manual_name,
                        "title": manual_title,
                        "location": manual_location,
                        "email": manual_email,
                        "summary": manual_summary,
                        "skills": [s.strip() for s in manual_skills.split(",") if s.strip()],
                        "experience": manual_experience,
                        "education": manual_education,
                    }
                else:
                    status_text.text("🔍 Scraping LinkedIn profile...")
                    profile_data = scrape_linkedin_profile(linkedin_url)
                
                progress_bar.progress(15)

                if not profile_data:
                    st.error("❌ Could not fetch profile. Try manual input.")
                    st.stop()

                # Show profile summary
                st.success(f"✅ Profile loaded: **{profile_data.get('name', 'Unknown')}** — {profile_data.get('title', '')}")

                # Step 2: Generate resumes
                status_text.text("🤖 Connecting to vLLM and generating resumes...")
                generator = ResumeGenerator(
                    vllm_url=vllm_url,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                resumes = []
                resume_types = generator.get_resume_types()
                
                for i, resume_type in enumerate(resume_types):
                    status_text.text(f"✍️ Generating resume {i+1}/10: {resume_type['name']}...")
                    resume_content = generator.generate_resume(profile_data, resume_type)
                    resumes.append({
                        "type": resume_type,
                        "content": resume_content
                    })
                    progress_bar.progress(15 + int((i + 1) / 10 * 75))
                    time.sleep(0.1)

                # Step 3: Store in session state
                st.session_state["resumes"] = resumes
                st.session_state["profile_data"] = profile_data
                
                progress_bar.progress(100)
                status_text.text("✅ All 10 resumes generated!")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure your vLLM server is running. See README for setup instructions.")
                progress_bar.empty()
                status_text.empty()

# Display results
if "resumes" in st.session_state:
    resumes = st.session_state["resumes"]
    profile_data = st.session_state["profile_data"]
    
    st.markdown("---")
    st.subheader(f"📄 10 Resume Versions for {profile_data.get('name', 'You')}")
    
    # Export all button
    col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 2])
    with col_exp1:
        if st.button("📥 Export All as ZIP"):
            with st.spinner("Packaging resumes..."):
                zip_bytes = export_resumes_to_pdf(resumes, profile_data)
                st.download_button(
                    label="⬇️ Download ZIP",
                    data=zip_bytes,
                    file_name=f"resumes_{profile_data.get('name','profile').replace(' ','_')}.zip",
                    mime="application/zip"
                )
    
    # Tabs for viewing
    tab_names = [f"{r['type']['emoji']} {r['type']['name']}" for r in resumes]
    tabs = st.tabs(tab_names)
    
    for i, (tab, resume) in enumerate(zip(tabs, resumes)):
        with tab:
            col_content, col_actions = st.columns([3, 1])
            
            with col_content:
                st.markdown(f"### {resume['type']['emoji']} {resume['type']['name']}")
                st.caption(resume['type']['description'])
                st.markdown("---")
                st.text_area(
                    "Resume Content",
                    value=resume['content'],
                    height=500,
                    key=f"resume_text_{i}",
                    label_visibility="collapsed"
                )
            
            with col_actions:
                st.markdown("### Actions")
                # Copy button
                st.code(resume['content'][:200] + "...", language=None)
                
                # Download individual
                st.download_button(
                    label="⬇️ Download .txt",
                    data=resume['content'],
                    file_name=f"resume_{resume['type']['name'].lower().replace(' ','_')}.txt",
                    mime="text/plain",
                    key=f"dl_{i}"
                )
                
                st.markdown("**Best for:**")
                for use in resume['type']['best_for']:
                    st.markdown(f"• {use}")
