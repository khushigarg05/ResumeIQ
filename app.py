import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="ResumeIQ | Candidate Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic_regression_tuned.pkl"

# ---------------- PREMIUM UI DESIGN ----------------
st.markdown("""
<style>
:root {
    --navy: #13213C;
    --blue: #315BE8;
    --light: #F3F6FC;
    --muted: #596B85;
    --border: #DCE5F2;
}

.stApp {
    background: #F3F6FC;
    color: #17243B;
}
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111D35;
}
[data-testid="stSidebar"] * {
    color: #F8FAFF !important;
}
.sidebar-brand {
    font-size: 27px;
    font-weight: 850;
    letter-spacing: -1px;
    color: #FFFFFF;
}
.sidebar-sub {
    color: #B9C9E8 !important;
    font-size: 13px;
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #142444 0%, #244CC0 62%, #547AF5 100%);
    border-radius: 22px;
    padding: 32px 34px;
    margin-bottom: 24px;
    box-shadow: 0 12px 30px rgba(31, 65, 145, .15);
}
.hero .eyebrow {
    color: #D6E2FF !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero h1 {
    color: #FFFFFF !important;
    font-size: clamp(30px, 4vw, 43px);
    font-weight: 850;
    line-height: 1.15;
    margin: 12px 0;
}
.hero p {
    color: #EDF2FF !important;
    font-size: 16px;
    line-height: 1.7;
    max-width: 760px;
}

/* Text visibility */
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp p, .stApp label, .stApp li,
[data-testid="stMarkdownContainer"] {
    color: #17243B;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: #596B85 !important;
}
[data-testid="stWidgetLabel"] p {
    color: #17243B !important;
    font-weight: 650;
}

/* Cards */
.panel {
    background: #FFFFFF;
    border: 1px solid #DCE5F2;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 16px;
}
.panel-title {
    font-size: 19px;
    font-weight: 800;
    color: #17243B !important;
}
.panel-description {
    font-size: 13px;
    color: #596B85 !important;
    line-height: 1.6;
}
.mini-card {
    background: #FFFFFF;
    border: 1px solid #DCE5F2;
    border-radius: 15px;
    padding: 17px;
    min-height: 112px;
}
.mini-label {
    color: #596B85 !important;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
}
.mini-value {
    color: #244CC0 !important;
    font-size: 24px;
    font-weight: 850;
    margin-top: 9px;
}

/* Form controls */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: #FFFFFF !important;
    color: #17243B !important;
    -webkit-text-fill-color: #17243B !important;
    border: 1px solid #B8C7DD !important;
    border-radius: 9px !important;
}
[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: #B8C7DD !important;
}
[data-baseweb="select"] *,
[data-baseweb="select"] input {
    color: #17243B !important;
    -webkit-text-fill-color: #17243B !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: #FFFFFF !important;
}
[data-baseweb="menu"] * {
    color: #17243B !important;
}
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #7392DE;
    border-radius: 13px;
    padding: 12px;
}
[data-testid="stFileUploader"] * {
    color: #17243B !important;
}

/* Buttons */
.stButton button, .stFormSubmitButton button,
[data-testid="stDownloadButton"] button {
    background: #315BE8 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 750 !important;
    min-height: 45px;
}
.stButton button *, .stFormSubmitButton button *,
[data-testid="stDownloadButton"] button * {
    color: #FFFFFF !important;
}
.stButton button:hover, .stFormSubmitButton button:hover {
    background: #2045C5 !important;
}

/* Metrics, alerts and expander */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE5F2;
    padding: 18px;
    border-radius: 15px;
}
[data-testid="stMetricLabel"] * {
    color: #596B85 !important;
}
[data-testid="stMetricValue"] * {
    color: #244CC0 !important;
}
[data-testid="stAlert"] {
    border-radius: 12px;
}
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #DCE5F2;
    border-radius: 12px;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #17243B !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    background: #E8EEF9 !important;
    border-radius: 9px 9px 0 0;
}
button[data-baseweb="tab"] *,
button[data-baseweb="tab"] p {
    color: #17243B !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: #D8E4FF !important;
    border-bottom: 3px solid #315BE8 !important;
}
button[data-baseweb="tab"][aria-selected="true"] * {
    color: #173B87 !important;
    font-weight: 800 !important;
}

/* Result panels */
.result-good {
    background: #E7F8EF;
    border: 1px solid #9BDDB8;
    border-left: 6px solid #168653;
    padding: 25px;
    border-radius: 15px;
}
.result-good h2 {
    color: #12663F !important;
}
.result-good p {
    color: #215F43 !important;
}
.result-review {
    background: #FFF1F0;
    border: 1px solid #F0B7B2;
    border-left: 6px solid #D94747;
    padding: 25px;
    border-radius: 15px;
}
.result-review h2 {
    color: #9F2727 !important;
}
.result-review p {
    color: #842B2B !important;
}
hr {
    border-color: #DCE5F2 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.info(
        "Confirm that logistic_regression_tuned.pkl is in the same "
        "folder as app.py and that the environment versions are compatible."
    )
    st.stop()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">🎯 ResumeIQ</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sidebar-sub">CANDIDATE INTELLIGENCE PLATFORM</p>',
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown("### Your workspace")
    st.markdown("🔎 &nbsp; Candidate screening")
    st.markdown("📄 &nbsp; Resume text extraction")
    st.markdown("📊 &nbsp; Prediction insights")
    st.divider()

    st.markdown("### Model inputs")
    st.markdown("• Experience")
    st.markdown("• Skills match")
    st.markdown("• Education")
    st.markdown("• Projects")
    st.markdown("• Resume length")
    st.markdown("• GitHub activity")

    st.divider()
    st.caption(
        "Academic prototype · Predictions require human review."
    )


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">AI · MACHINE LEARNING · TALENT ANALYTICS</div>
    <h1>Find potential.<br>Screen with insight.</h1>
    <p>
        Evaluate structured candidate information using your trained
        machine-learning pipeline. Review the details before generating
        a prediction.
    </p>
</div>
""", unsafe_allow_html=True)

# Summary cards
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="mini-card">
        <div class="mini-label">01 · Provide</div>
        <div class="mini-value">Candidate data</div>
        <div class="panel-description">Enter six model features.</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="mini-card">
        <div class="mini-label">02 · Validate</div>
        <div class="mini-value">Review inputs</div>
        <div class="panel-description">Confirm values before analysis.</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="mini-card">
        <div class="mini-label">03 · Predict</div>
        <div class="mini-value">ML result</div>
        <div class="panel-description">Review the model estimate.</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ---------------- TABS ----------------
tab_assess, tab_resume, tab_about = st.tabs([
    "✦  Candidate Assessment",
    "📄  Resume Text",
    "ⓘ  Model Information"
])


# ---------------- CANDIDATE ASSESSMENT ----------------
with tab_assess:
    st.markdown("## Candidate assessment")
    st.markdown(
        '<p style="color:#596B85">Complete all six fields using '
        'verified candidate information. The model uses these exact '
        'dataset features.</p>',
        unsafe_allow_html=True
    )

    with st.form("candidate_evaluation_form"):
        st.markdown("### 👤 Professional profile")
        col1, col2 = st.columns(2, gap="large")

        with col1:
            experience = st.number_input(
                "Years of experience",
                min_value=0,
                max_value=50,
                value=2,
                step=1,
                help="Total relevant work experience in years. Use 0 for a fresher."
            )

            education = st.selectbox(
                "Highest education level",
                ["Bachelors", "Masters", "High School", "Phd"],
                help="Select a category used in the training dataset."
            )

            projects = st.number_input(
                "Number of projects",
                min_value=0,
                max_value=100,
                value=3,
                step=1,
                help="Number of relevant academic, personal, or professional projects."
            )

        with col2:
            skills_score = st.slider(
                "Skills match score (%)",
                min_value=0,
                max_value=100,
                value=75,
                step=1,
                help="A skills-match score calculated against the role requirements. "
                     "This prototype does not calculate it automatically."
            )

            resume_length = st.number_input(
                "Resume length (words)",
                min_value=1,
                max_value=10000,
                value=500,
                step=50,
                help="Approximate word count of the resume."
            )

            github_activity = st.number_input(
                "GitHub activity score",
                min_value=0,
                max_value=100000,
                value=100,
                step=10,
                help="Use the same activity metric definition used by your dataset."
            )

        st.divider()
        st.markdown("### 💼 Role context (optional)")
        job_title = st.text_input(
            "Job title",
            placeholder="e.g. Junior Python Developer"
        )
        job_description = st.text_area(
            "Job description / required skills",
            placeholder="Enter the key responsibilities and skills required for the role...",
            height=100
        )
        st.caption(
            "Job title and description are collected for context only. "
            "The current trained model does not directly process these text fields."
        )

        st.divider()
        confirmed = st.checkbox(
            "I have checked these values and confirm they represent the candidate's information.",
            value=False
        )

        submitted = st.form_submit_button(
            "✦  Analyze Candidate",
            use_container_width=True
        )

    if submitted:
        if not confirmed:
            st.error(
                "Please confirm the candidate information before running the prediction."
            )
        else:
            candidate = pd.DataFrame([{
                "years_experience": experience,
                "skills_match_score": skills_score,
                "education_level": education,
                "project_count": projects,
                "resume_length": resume_length,
                "github_activity": github_activity
            }])

            expected_features = [
                "years_experience",
                "skills_match_score",
                "education_level",
                "project_count",
                "resume_length",
                "github_activity"
            ]

            try:
                # Keep the same feature names and ordering as training.
                candidate = candidate[expected_features]
                prediction = model.predict(candidate)[0]

                probability = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(candidate)[0]
                    classes = list(model.classes_)
                    if 1 in classes:
                        probability = float(
                            probabilities[classes.index(1)]
                        )

                st.divider()
                st.markdown("## 📊 Candidate analysis")

                if prediction == 1:
                    st.markdown("""
                    <div class="result-good">
                        <h2>✓ Model prediction: Shortlisted</h2>
                        <p>The model predicts the positive shortlisting class.
                        Review the candidate's evidence and requirements before
                        making any decision.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-review">
                        <h2>Model prediction: Not Shortlisted</h2>
                        <p>The model predicts the negative shortlisting class.
                        This output should not be treated as a final rejection
                        or as proof that the candidate lacks ability.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                metric1, metric2 = st.columns(2)

                with metric1:
                    if probability is not None:
                        st.metric(
                            "Estimated probability of shortlisting",
                            f"{probability:.1%}"
                        )
                    else:
                        st.metric("Prediction generated", "Yes")

                with metric2:
                    st.metric(
                        "Candidate features checked",
                        f"{len(candidate.columns)} / 6"
                    )

                if probability is not None:
                    st.progress(probability)

                st.markdown("### Candidate profile summary")

                # -------- WHY DID THE MODEL MAKE THIS PREDICTION? --------
                st.markdown("### 🔍 Why this prediction?")

                st.markdown(
                    "These insights show which input features pushed the "
                    "Logistic Regression model toward or away from the "
                    "shortlisting prediction. They are not proof of the "
                    "candidate's actual strengths or weaknesses."
                )

                try:
                    preprocessor = model.named_steps["preprocessor"]
                    classifier = model.named_steps["classifier"]

                    if hasattr(classifier, "coef_"):
                        transformed = preprocessor.transform(candidate)

                        if hasattr(transformed, "toarray"):
                            transformed = transformed.toarray()

                        feature_names = preprocessor.get_feature_names_out()
                        coefficients = classifier.coef_[0]

                        # Contribution to the model's decision score.
                        contributions = transformed[0] * coefficients

                        explanation_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Contribution": contributions
                        })

                        negative = (
                            explanation_df[
                                explanation_df["Contribution"] < 0
                            ]
                            .sort_values("Contribution")
                        )

                        positive = (
                            explanation_df[
                                explanation_df["Contribution"] > 0
                            ]
                            .sort_values(
                                "Contribution",
                                ascending=False
                            )
                        )

                        left, right = st.columns(2, gap="large")

                        with left:
                            st.markdown("#### ⚠️ Factors lowering the score")

                            if negative.empty:
                                st.success(
                                    "No negative feature contributions "
                                    "were found for this input."
                                )
                            else:
                                for _, row in negative.head(5).iterrows():
                                    feature = row["Feature"].lower()
                                    name = row["Feature"].split("__")[-1]

                                    if "skills_match_score" in feature:
                                        tip = (
                                            "Review the score against the "
                                            "actual job requirements. Identify "
                                            "relevant skills to develop or "
                                            "demonstrate with projects."
                                        )
                                    elif "years_experience" in feature:
                                        tip = (
                                            "Highlight relevant internships, "
                                            "practical work, and measurable "
                                            "results. Do not exaggerate experience."
                                        )
                                    elif "project_count" in feature:
                                        tip = (
                                            "Showcase relevant projects with "
                                            "clear problem statements, tools, "
                                            "your contribution, and outcomes."
                                        )
                                    elif "github_activity" in feature:
                                        tip = (
                                            "If this metric accurately reflects "
                                            "the candidate's work, highlight "
                                            "relevant repositories and explain "
                                            "their contributions."
                                        )
                                    elif "resume_length" in feature:
                                        tip = (
                                            "Review resume clarity and relevance. "
                                            "Prioritize useful evidence rather "
                                            "than simply increasing word count."
                                        )
                                    elif "education_level" in feature:
                                        tip = (
                                            "This education category has a "
                                            "negative learned association in "
                                            "the model. Check actual job criteria "
                                            "and possible dataset bias; do not "
                                            "treat this as proof of a deficiency."
                                        )
                                    else:
                                        tip = (
                                            "Review this feature and confirm "
                                            "that it accurately represents "
                                            "the candidate."
                                        )

                                    with st.container(border=True):
                                        st.markdown(f"**{name}**")
                                        st.caption(
                                            f"Contribution: "
                                            f"{row['Contribution']:.3f}"
                                        )
                                        st.write(tip)

                        with right:
                            st.markdown("#### ✅ Factors supporting the score")

                            if positive.empty:
                                st.info(
                                    "No positive feature contributions "
                                    "were found for this input."
                                )
                            else:
                                for _, row in positive.head(5).iterrows():
                                    name = row["Feature"].split("__")[-1]

                                    with st.container(border=True):
                                        st.markdown(f"**{name}**")
                                        st.caption(
                                            f"Contribution: "
                                            f"+{row['Contribution']:.3f}"
                                        )
                                        st.write(
                                            "This feature pushes the model's "
                                            "decision score toward the "
                                            "shortlisted class for this input."
                                        )

                        st.caption(
                            "Contribution values are in the model's transformed "
                            "decision-score space. Positive values support the "
                            "shortlisted class; negative values oppose it. "
                            "These values are not percentages or causal explanations."
                        )

                    else:
                        st.info(
                            "Detailed coefficient-based explanations are "
                            "unavailable for this model type."
                        )

                except Exception as explanation_error:
                    st.warning(
                        "The prediction was generated, but its explanation "
                        f"could not be calculated: {explanation_error}"
                    )
                summary1, summary2, summary3 = st.columns(3)

                summary1.metric("Experience", f"{experience} years")
                summary2.metric("Skills match", f"{skills_score}%")
                summary3.metric("Education", education)

                with st.expander("View all features sent to the model"):
                    st.dataframe(candidate, use_container_width=True)

                    st.download_button(
                        "Download candidate feature summary",
                        data=candidate.to_csv(index=False),
                        file_name="candidate_feature_summary.csv",
                        mime="text/csv"
                    )

                st.caption(
                    "The displayed probability is a model output, not necessarily "
                    "a calibrated measure of real-world hiring success. This "
                    "prototype should support—not replace—human review. Validate "
                    "the dataset labels, model performance, and potential bias "
                    "before any real hiring use."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")
                st.info(
                    "Check the saved model, feature names, education categories, "
                    "and package versions."
                )


# ---------------- PDF / RESUME TEXT ----------------
with tab_resume:
    st.markdown("## Resume text extraction")
    st.markdown(
        "Upload a PDF to inspect selectable text before manually entering "
        "verified candidate details."
    )

    uploaded_file = st.file_uploader(
        "Upload candidate resume (PDF)",
        type=["pdf"],
        help="Text-based PDFs are supported. Scanned PDFs may require OCR."
    )

    if uploaded_file is not None:
        try:
            from pypdf import PdfReader

            reader = PdfReader(uploaded_file)
            extracted_text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

            if extracted_text.strip():
                st.success(
                    f"Extracted text from {len(reader.pages)} page(s)."
                )
                st.text_area(
                    "Review extracted resume text",
                    value=extracted_text,
                    height=320
                )

                st.download_button(
                    "Download extracted text",
                    data=extracted_text,
                    file_name="resume_text.txt",
                    mime="text/plain"
                )
            else:
                st.warning(
                    "No selectable text was found. This may be a scanned PDF. "
                    "OCR support is not included in this version."
                )

        except ImportError:
            st.error(
                "PDF support is not installed. Run %pip install pypdf "
                "in your Jupyter Notebook, then restart the app."
            )
        except Exception as error:
            st.error(f"Unable to read this PDF: {error}")

    st.info(
        "Important: PDF text is not automatically converted into the six "
        "model features. Review the resume and enter verified values in "
        "Candidate Assessment."
    )


# ---------------- MODEL INFORMATION ----------------
with tab_about:
    st.markdown("## How ResumeIQ works")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">01 · Structured input</div>
        <p class="panel-description">
        Six features are collected using the same column names expected
        by the trained pipeline.
        </p>
        <hr>
        <div class="panel-title">02 · Preprocessing and prediction</div>
        <p class="panel-description">
        The saved scikit-learn pipeline applies its fitted preprocessing
        and classification steps.
        </p>
        <hr>
        <div class="panel-title">03 · Results for human review</div>
        <p class="panel-description">
        The app displays the predicted class and, if available, the model's
        estimated probability for the positive class.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Features used by the model")

    feature_info = pd.DataFrame({
        "Dataset feature": [
            "years_experience",
            "skills_match_score",
            "education_level",
            "project_count",
            "resume_length",
            "github_activity"
        ],
        "Information required": [
            "Relevant experience in years",
            "Skills match score from 0 to 100",
            "High School, Bachelors, Masters, or Phd",
            "Count of relevant projects",
            "Resume word count",
            "Dataset-compatible GitHub activity metric"
        ]
    })

    st.dataframe(feature_info, use_container_width=True, hide_index=True)

    st.warning(
        "The model learns patterns from its training labels. It does not "
        "establish a candidate's true potential or guarantee future job performance."
    )

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#596B85;font-size:12px;">'
    'ResumeIQ · Academic ML Prototype · Human-reviewed candidate assessment'
    '</p>',
    unsafe_allow_html=True
)
