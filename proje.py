import streamlit as st
import numpy as np
import pandas as pd
import io

# --- Page Config ---
st.set_page_config(
    page_title="🌍 Decision Support System For Decarbonization Activities In Energy Intensive Industries",
    layout="centered",
)

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
    "Carbon Capture and Storage ",
    "Alternative Fuel Sources",
    "Reuse Waste Heat",
    "Sustainable Material Selection and Recycling",
    "Digitalization and Industry 4.0 Applications",
]

SECTORS = [
    "metal industries",
    "cement industries",
    "chemical production industries",
    "oil and gas industries",
    "critical mineral industries",
]

# --- Constants: ADKAR ---
ADKAR_FACTORS = ["Awareness", "Desire", "Knowledge", "Ability", "Reinforcement"]

ADKAR_QUESTIONS = {
    "Awareness": [
        "The front-line workers know what kind of environmental and financial risks we will face if the company fails to decarbonize.",
        "Middle managers can articulate how decarbonization relates to company business strategy and performance KPIs",
        "Top management continually expresses concern about the need to meet future regulatory or customer climate goals.",
        "Company managers know which decarbonization technologies are the most relevant to processes.",
        "A compelling case for change has been articulated in company-wide communications (town-halls, newsletters, intranet).",
    ],
    "Desire": [
        "Employees are motivated to help the company meet its decarbonization goals.",
        "Leaders in the department are willing to invest in budget and staff time to promote sustainability, even if short-term costs are higher.",
        "Company’s incentive systems (bonuses, recognition, career paths) reward progress on carbon-reduction targets.",
        "Employees feel that adopting greener practices will improve job security and the company’s public image.",
        "Informal influencers (trusted peers, union reps, technical specialists) openly advocate for the transition.",
    ],
    "Knowledge": [
        "Staff who operate key equipment have been trained on energy-efficient operating procedures.",
        "The company understands the data that has to be collect and report to track CO₂ savings and compliance obligations.",
        "Clear standard operating procedures exist for integrating new low-carbon technologies into production.",
        "Employees are aware of external standards or frameworks that have to be meet.",
        "The company provides easy access to reference guides, job aids, or experts who can answer technical questions.",
    ],
    "Ability": [
        "The company have sufficient skilled personnel to implement and maintain new decarbonization technologies",
        "The necessary infrastructure (power supply, process controls, digital monitoring) is in place or budgeted.",
        "Project teams have the authority to resolve issues rapidly without excessive bureaucracy.",
        "Workloads allow time for employees to practise new skills without compromising safety or output.",
        "Coaching, troubleshooting support, or external consultants are available when problems arise.",
    ],
    "Reinforcement": [
        "KPIs related to carbon reduction will be reviewed at least quarterly and will affect decision-making.",
        "Best practice and learning points from the pilot projects will be reported throughout the company.",
        "With continuous improvement or audit, saved heights of CO2 emissions are automatically measured.",
        "High-performing teams or sites will receive visible recognition for decarbonization achievements.",
        "New hires and contractors will receive onboarding that emphasises the company’s sustainability commitments.",
    ],
}

# Discrete AHP scale
SCALE = [-9, -7, -5, -3, 1, 3, 5, 7, 9]

# ===================== AHP Helper Functions ===================== #

def pairwise_matrix(items, key_prefix):
    """Render pairwise comparison sliders and return numpy matrix."""
    n = len(items)
    mat = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            key = f"{key_prefix}_{i}_{j}"
            c1, c2, c3 = st.columns([4, 6, 4])
            with c1:
                st.markdown(
                    f"<div style='text-align:center;border:1px solid #ccc;padding:4px;border-radius:6px;font-size:14px;'>{items[i]}</div>",
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
                    f"<div style='text-align:center;border:1px solid #ccc;padding:4px;border-radius:6px;font-size:14px;'>{items[j]}</div>",
                    unsafe_allow_html=True,
                )
    return mat


def priority_vector(mat):
    """Calculate priority vector via eigenvector method (approx by averaging normalised columns)."""
    return np.mean(mat / np.sum(mat, axis=0), axis=1)


def consistency_ratio(mat, pv):
    n = len(pv)
    lam_max = (np.dot(mat, pv) / pv).mean()
    ci = (lam_max - n) / (n - 1)
    RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}.get(n, 1.49)
    return ci / RI if RI else 0

# ---------- NEW: Inconsistency Diagnostics ---------- #

def _inconsistency_details(mat: np.ndarray, weights: np.ndarray, items: list[str], top_n: int = 3):
    """Return list of the top_n pairwise comparisons causing highest inconsistency."""
    n = len(items)
    deviations = []
    for i in range(n):
        for j in range(i + 1, n):
            expected = weights[i] / weights[j] if weights[j] else np.inf
            actual = mat[i, j]
            err = abs(np.log(actual) - np.log(expected))
            deviations.append((err, i, j, actual, expected))
    deviations.sort(reverse=True, key=lambda x: x[0])
    return deviations[:top_n]


def show_inconsistency_warning(cr: float, mat: np.ndarray, wts: np.ndarray, items: list[str]):
    """Streamlit warning guiding user to specific sliders when CR is too high."""
    if cr <= 0.10:
        return

    issues = _inconsistency_details(mat, wts, items)
    lines = [
        "⚠️ **High Consistency Ratio detected (CR > 0.10).**",
        "These comparisons contribute most to the inconsistency:",
    ]
    for _, i, j, _actual, _expected in issues:
        lines.append(f"- **{items[i]} ↔ {items[j]}** - Please review your judgment.")
    lines.append("\nAdjust these sliders first to improve consistency.")
    st.warning("\n".join(lines))

# ===================== Session State ===================== #
if "selected_sector" not in st.session_state:
    st.session_state.selected_sector = None

# ===================== Sector Page ===================== #

def run_sector(sector: str):
    st.sidebar.button(
        "⬅️ Back to Sector Menu", on_click=lambda: st.session_state.update({"selected_sector": None})
    )
    st.title(f"🌍 Decision Support System For Decarbonization Activities In Energy Intensive Industries – {sector.title()} ")

    # ① Criteria Comparison
    st.header("① Compare Criteria")
    with st.expander("Compare Criteria - Click to Expand", expanded=True):
        crit_mat = pairwise_matrix(CRITERIA, f"{sector}_crit")
        crit_w = priority_vector(crit_mat)
        crit_cr = consistency_ratio(crit_mat, crit_w)
        st.dataframe(pd.DataFrame({"Criteria": CRITERIA, "Weight": crit_w}))
        st.markdown(f"**Consistency Ratio (CR):** `{crit_cr:.3f}`")
        show_inconsistency_warning(crit_cr, crit_mat, crit_w, CRITERIA)

    # ② Alternative Evaluation
    st.header("② Compare Alternatives")
    alt_results = {}
    for crit in CRITERIA:
        with st.expander(crit):
            m = pairwise_matrix(ALTERNATIVES, f"{sector}_{crit}")
            w = priority_vector(m)
            alt_results[crit] = w
            cr = consistency_ratio(m, w)
            st.dataframe(pd.DataFrame({"Alternative": ALTERNATIVES, "Weight": w}))
            st.markdown(f"CR: `{cr:.3f}`")
            show_inconsistency_warning(cr, m, w, ALTERNATIVES)

    # ③ Best Alternative
    st.header("③ Best Alternative for Sector")
    scores = np.zeros(len(ALTERNATIVES))
    for i, c in enumerate(CRITERIA):
        scores += crit_w[i] * alt_results[c]
    best_idx = int(np.argmax(scores))
    best_alt = ALTERNATIVES[best_idx]
    st.success(f"✅ {sector.title()} best alternative → **{best_alt}** (score {scores[best_idx]:.3f})")

    st.dataframe(pd.DataFrame({"Alternative": ALTERNATIVES, "Score": scores}))

    # ④ ADKAR Readiness Survey
    st.header("④ ADKAR Readiness Survey – 25 Questions")
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

        overall = float(np.mean(list(fac_norm.values())))

        def readiness_text(v: float):
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
            emoji = "🟢" if fac_norm[fac] >= 0.60  else "🔴"
            cols[i].metric(fac, f"{fac_norm[fac]:.0%}", emoji)

        low = [f for f, v in fac_norm.items() if v < 0.6]
        if low:
            st.markdown("### Suggested Actions")
            for fac in low:
                st.write(f"**{fac}:**")
                for sug in ADKAR_QUESTIONS[fac][:2]:
                    st.markdown(f"- {sug}")

        # ⑤ Download Consolidated Excel Report
        st.header("⑤ Download Sector Report")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            pd.DataFrame({"Criteria": CRITERIA, "Weight": crit_w}).to_excel(
                writer, sheet_name="Criteria Weights", index=False
            )
            pd.DataFrame({"Alternative": ALTERNATIVES, "Score": scores}).to_excel(
                writer, sheet_name="Alternative Scores", index=False
            )
            pd.DataFrame(
                {
                    "Factor": ADKAR_FACTORS,
                    "Readiness": [fac_norm[f] for f in ADKAR_FACTORS],
                }
            ).to_excel(writer, sheet_name="ADKAR Readiness", index=False)
            pd.DataFrame(
                {"Metric": ["Overall Readiness"], "Value": [overall]}
            ).to_excel(writer, sheet_name="ADKAR Summary", index=False)

        st.download_button(
            "📥 Download Excel Report",
            data=buf,
            file_name=f"AHP_ADKAR_{sector.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ===================== Main Routing ===================== #
if st.session_state.selected_sector is None:
    st.title("🌍 Decision Support System For Decarbonization Activities In Energy Intensive Industries")
    st.markdown(
        """
        <div style='text-align:center;'>Select a sector to begin. You can return to this menu via the sidebar.</div>
        """,
        unsafe_allow_html=True,
    )
    for s in SECTORS:
        if st.button(s.title()):
            st.session_state.selected_sector = s    
            st.rerun()
else:
    run_sector(st.session_state.selected_sector)
