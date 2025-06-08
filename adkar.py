import streamlit as st
import numpy as np
import pandas as pd
import io

# --- Page Config ---
st.set_page_config(page_title="AHP Multi-Sector Evaluator", layout="centered")

# --- Inject Custom Slider / SelectSlider Style ---
st.markdown(
    """
    <style>
    input[type=range]::-webkit-slider-runnable-track {background:linear-gradient(to right,#ff4b4b 50%,#31333f 50%);} 
    input[type=range]::-moz-range-track {background:linear-gradient(to right,#ff4b4b 50%,#31333f 50%);} 
    .stSelectSlider > div[data-baseweb="slider"] > div {width:100% !important;padding-left:10px;padding-right:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Constants: AHP ---
CRITERIA = [
    "Carbon Reduction Potential and Environmental Co-Benefits",
    "Economic Feasibility",
    "Technological Readiness and Implementation Feasibility",
    "Scalability and Long-Term Sustainability",
    "Policy Alignment",
    "Social Acceptance",
]

ALTERNATIVES = [
    "CCS / CCUS carbon capture and storage (filtre)",
    "Fuel Switching / Alternative Fuel Sources",
    "Reuse Waste Heat",
    "Sustainable Material Selection and Recycling",
    "Digitalization and Industry 4.0 Applications",
]

SECTORS = [
    "metal industries",
    "cement",
    "chemical production",
    "oil and gas",
    "critical mineral industry",
]

# --- Constants: ADKAR ---
ADKAR_FACTORS = ["Awareness", "Desire", "Knowledge", "Ability", "Reinforcement"]

ADKAR_QUESTIONS = {
    "Awareness": [
        "Employees understand why decarbonization is critical.",
        "Environmental targets have been clearly communicated.",
        "Leadership highlights risks of not switching fuel.",
        "Staff recognise regulatory pressure on emissions.",
        "Timeline and urgency are commonly understood.",
    ],
    "Desire": [
        "Employees are motivated to support the change.",
        "Middle management actively champions the switch.",
        "Stakeholders see personal benefits in this change.",
        "There is visible enthusiasm for pilot tests.",
        "People are willing to invest time in learning new tech.",
    ],
    "Knowledge": [
        "Teams know technical steps for retrofit.",
        "Safety procedures are documented.",
        "Operations know hydrogen‑handling requirements.",
        "Training materials are prepared.",
        "Teams know success metrics for the new fuel.",
    ],
    "Ability": [
        "Staff have the necessary skills for implementation.",
        "Budget / resources are secured.",
        "Company has prior similar project experience.",
        "Processes can absorb the technological change.",
        "Integration can occur without major disruption.",
    ],
    "Reinforcement": [
        "Incentives align with sustaining the change.",
        "KPIs reflect the new process.",
        "Follow‑up audits are planned.",
        "Leadership will publicly recognise success.",
        "Policies will embed the new practices.",
    ],
}

# Discrete AHP scale
SCALE = [-9, -7, -5, -3, 1, 3, 5, 7, 9]

# --- Helper Functions: AHP ---

def pairwise_matrix(items, key_prefix):
    n = len(items)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{key_prefix}_{i}_{j}"
            c1, c2, c3 = st.columns([4, 6, 4])
            with c1:
                st.markdown(
                    f"<div style='text-align:center;border:1px solid #ccc;padding:4px;border-radius:6px;font-size:14px;'>" + items[i] + "</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                sel = st.select_slider(
                    " ",
                    options=SCALE,
                    value=1,
                    key=key,
                    help="Negative → left more important, positive → right more important, 1 → equal importance.",
                )
                if sel == 1:
                    mat[i, j] = mat[j, i] = 1
                elif sel > 1:
                    mat[i, j] = 1 / sel
                    mat[j, i] = sel
                else:
                    w = abs(sel)
                    mat[i, j] = w
                    mat[j, i] = 1 / w
            with c3:
                st.markdown(
                    f"<div style='text-align:center;border:1px solid #ccc;padding:4px;border-radius:6px;font-size:14px;'>" + items[j] + "</div>",
                    unsafe_allow_html=True,
                )
    return mat


def priority_vector(mat):
    return np.mean(mat / np.sum(mat, axis=0), axis=1)


def consistency_ratio(mat, pv):
    n = len(pv)
    lam_max = (np.dot(mat, pv) / pv).mean()
    ci = (lam_max - n) / (n - 1)
    RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}.get(n, 1.49)
    return ci / RI if RI else 0

# --- Session State ---
if "selected_sector" not in st.session_state:
    st.session_state.selected_sector = None

# --- Sector Evaluation Page ---

def run_sector(sector: str):
    st.sidebar.button("⬅️ Back to Sector Menu", on_click=lambda: st.session_state.update({"selected_sector": None}))
    st.title(f"🌍 AHP Evaluation – {sector.title()} Sector")

    # ① Criteria Comparison
    st.header("① Compare Evaluation Criteria")
    with st.expander("Compare Criteria - Click to Expand", expanded=True):
        crit_mat = pairwise_matrix(CRITERIA, f"{sector}_crit")
        crit_w = priority_vector(crit_mat)
        crit_cr = consistency_ratio(crit_mat, crit_w)
        st.dataframe(pd.DataFrame({"Criteria": CRITERIA, "Weight": crit_w}))
        st.markdown(f"**Consistency Ratio (CR):** `{crit_cr:.3f}`")
        if crit_cr > 0.1:
            st.warning("⚠️ High inconsistency – consider revising judgments.")

    # ② Alternative Evaluation
    st.header("② Alternative Evaluation")
    alt_results = {}
    for crit in CRITERIA:
        with st.expander(crit):
            m = pairwise_matrix(ALTERNATIVES, f"{sector}_{crit}")
            w = priority_vector(m)
            alt_results[crit] = w
            cr = consistency_ratio(m, w)
            st.dataframe(pd.DataFrame({"Alternative": ALTERNATIVES, "Weight": w}))
            st.markdown(f"CR: `{cr:.3f}`")
            if cr > 0.1:
                st.warning("⚠️ Inconsistent comparison – adjust values.")

    # ③ Best Alternative
    st.header("③ Best Alternative for Sector")
    scores = np.zeros(len(ALTERNATIVES))
    for i, c in enumerate(CRITERIA):
        scores += crit_w[i] * alt_results[c]
    best_idx = int(np.argmax(scores))
    best_alt = ALTERNATIVES[best_idx]
    st.success(f"✅ {sector.title()} best alternative → **{best_alt}** (score {scores[best_idx]:.3f})")

    st.dataframe(pd.DataFrame({"Alternative": ALTERNATIVES, "Score": scores}))

    # ④ Download Report
    st.header("④ Download Sector Report")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        pd.DataFrame({"Criteria": CRITERIA, "Weight": crit_w}).to_excel(writer, sheet_name="Criteria Weights", index=False)
        pd.DataFrame({"Alternative": ALTERNATIVES, "Score": scores}).to_excel(writer, sheet_name="Alternative Scores", index=False)
    st.download_button("📥 Download Excel Report", data=buf, file_name=f"AHP_{sector.replace(' ', '_')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ⑤ ADKAR Readiness Survey
    st.header("⑤ ADKAR Readiness Survey – 25 Questions")
    st.markdown(
        f"*Chosen alternative:* **{best_alt}** – rate readiness (1 = Strongly Disagree … 5 = Strongly Agree)"
    )

    responses = {}
    for fac in ADKAR_FACTORS:
        st.markdown(f"#### {fac}")
        for q_idx, q in enumerate(ADKAR_QUESTIONS[fac]):
            key = f"{sector}_{fac}_{q_idx}"
            responses[key] = st.slider(q, 1, 5, 3, key=key)

    if st.button("Calculate Readiness", key=f"calc_{sector}"):
        fac_norm = {}
        for fac in ADKAR_FACTORS:
            vals = [responses[f"{sector}_{fac}_{i}"] for i in range(5)]
            fac_norm[fac] = np.mean([(v - 1) / 4 for v in vals])

        overall = np.mean(list(fac_norm.values()))

        def readiness_text(v):
            if v >= 0.75:
                return "**High readiness** – organisation is well‑prepared."
            elif v >= 0.6:
                return "**Moderate readiness** – some gaps need addressing."
            else:
                return "**Low readiness** – significant preparation required."

        st.subheader("Readiness Result")
        st.markdown(f"Overall readiness: **{overall:.0%}** – {readiness_text(overall)}")

        cols = st.columns(5)
        for i, fac in enumerate(ADKAR_FACTORS):
            emoji = "🟢" if fac_norm[fac] >= 0.75 else ("🟡" if fac_norm[fac] >= 0.6 else "🔴")
            cols[i].metric(fac, f"{fac_norm[fac]:.0%}", emoji)

        low = [f for f, v in fac_norm.items() if v < 0.6]
        if low:
            st.markdown("### Suggested Actions")
            for fac in low:
                st.write(f"**{fac}:**")
                for sug in ADKAR_QUESTIONS[fac][:2]:
                    st.markdown(f"- {sug}")

# --- Main Routing ---
if st.session_state.selected_sector is None:
    st.title("🌍 AHP Multi-Sector Decarbonization Evaluator")
    st.markdown(
        """
        <div style='text-align:center;'>Select a sector to begin. You can return to this menu via the sidebar.</div>
        """,
        unsafe_allow_html=True,
    )
    for s in SECTORS:
        if st.button(s.title()):
            st.session_state.selected_sector = s
            st.experimental_rerun()
else:
    run_sector(st.session_state.selected_sector)
