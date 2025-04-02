import streamlit as st
import numpy as np
import pandas as pd
import io

# --- Page Config ---
st.set_page_config(page_title="AHP Multi-Sector Evaluator", layout="centered")

# --- Inject Custom Slider Style ---
st.markdown("""
    <style>
    /* Slider track coloring */
    input[type=range]::-webkit-slider-runnable-track {
        background: linear-gradient(to right, #ff4b4b 50%, #31333f 50%);
    }
    input[type=range]::-moz-range-track {
        background: linear-gradient(to right, #ff4b4b 50%, #31333f 50%);
    }
    .stSlider > div[data-baseweb="slider"] > div {
        width: 100% !important;
        padding-left: 10px;
        padding-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
CRITERIA = [
    "Carbon Reduction Potential and Environmental Co-Benefits",
    "Economic Feasibility",
    "Technological Readiness and Implementation Feasibility",
    "Scalability and Long-Term Sustainability",
    "Policy Alignment",
    "Social Acceptance"
]

ALTERNATIVES = [
    "CCS / CCUS carbon capture and storage (filtre)",
    "Fuel Switching / Alternative Fuel Sources",
    "Reuse Waste Heat",
    "Sustainable Material Selection and Recycling",
    "Digitalization and Industry 4.0 Applications"
]

SECTORS = [
    "metal industries",
    "cement",
    "chemical production",
    "oil and gas",
    "critical mineral industry"
]

# --- AHP Logic ---
def pairwise_matrix(items, session_key):
    n = len(items)
    matrix = np.ones((n, n))
    scale = list(range(-9, 10))  # -9 to 9
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{session_key}_{i}_{j}"
            col1, col2, col3 = st.columns([4, 6, 4])
            with col1:
                st.markdown(
                    f"<div style='text-align:center; border:1px solid #ccc; padding:4px; border-radius:6px; font-size:14px;'>"
                    f"{items[i]}</div>",
                    unsafe_allow_html=True
                )
            with col2:
                selection = st.slider(
                    label=" ",
                    min_value=-9,
                    max_value=9,
                    value=0,
                    step=1,
                    key=key,
                    help="0 = equally important, positive = right is more important, negative = left is more important"
                )
                if selection == 0:
                    matrix[i][j] = 1
                    matrix[j][i] = 1
                elif selection > 0:
                    matrix[i][j] = 1 / selection
                    matrix[j][i] = selection
                else:  # selection < 0
                    matrix[i][j] = abs(selection)
                    matrix[j][i] = 1 / abs(selection)
            with col3:
                st.markdown(
                    f"<div style='text-align:center; border:1px solid #ccc; padding:4px; border-radius:6px; font-size:14px;'>"
                    f"{items[j]}</div>",
                    unsafe_allow_html=True
                )
    return matrix


def normalize_matrix(matrix):
    col_sum = np.sum(matrix, axis=0)
    return matrix / col_sum


def calculate_priority_vector(matrix):
    norm = normalize_matrix(matrix)
    return np.mean(norm, axis=1)


def consistency_ratio(matrix, priority_vector):
    n = len(priority_vector)
    lamda_max = np.sum(np.dot(matrix, priority_vector) / priority_vector) / n
    ci = (lamda_max - n) / (n - 1)
    RI_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = RI_dict.get(n, 1.49)
    cr = ci / ri if ri != 0 else 0
    return cr


# --- Streamlit UI ---
st.title("🌍 AHP Multi-Sector Decarbonization Evaluator")
st.markdown("""
<div style='text-align: center;'>
    Welcome to the AHP application for evaluating decarbonization strategies across sectors. 
    You'll walk through four steps to build a decision framework and receive a downloadable report.
</div>
""", unsafe_allow_html=True)

st.header("① Compare Evaluation Criteria")
with st.expander("Compare Criteria - Click to Expand"):
    st.write("Provide pairwise comparisons between the six evaluation criteria:")
    criteria_matrix = pairwise_matrix(CRITERIA, "criteria")
    criteria_weights = calculate_priority_vector(criteria_matrix)
    criteria_cr = consistency_ratio(criteria_matrix, criteria_weights)

    st.subheader("Criteria Weights")
    st.dataframe(pd.DataFrame({"Criteria": CRITERIA, "Weight": criteria_weights}))
    st.markdown(f"**Consistency Ratio (CR):** `{criteria_cr:.3f}`")
    if criteria_cr > 0.1:
        st.warning("⚠️ The consistency ratio is high. Consider revisiting your judgments.")

st.header("② Sector-Wise Evaluation of Alternatives")
st.markdown("<div style='text-align:center;'>Evaluate alternatives for <strong>all sectors</strong> below:</div>", unsafe_allow_html=True)

if 'sector_results' not in st.session_state:
    st.session_state.sector_results = {}

sector_best_alternatives = {}

for sector in SECTORS:
    st.subheader(f"{sector.title()}")
    for criterion in CRITERIA:
        with st.expander(f"{criterion}", expanded=False):
            matrix = pairwise_matrix(ALTERNATIVES, f"{sector}_{criterion}")
            weights = calculate_priority_vector(matrix)
            cr = consistency_ratio(matrix, weights)
            df = pd.DataFrame({"Alternative": ALTERNATIVES, "Weight": weights})
            st.dataframe(df)
            st.markdown(f"**Consistency Ratio (CR):** `{cr:.3f}`")
            if cr > 0.1:
                st.warning("⚠️ Inconsistent comparison. Try adjusting the values.")
            st.session_state.sector_results[(sector, criterion)] = weights

st.header("③ Best Alternative per Sector")
sector_final_scores = {}
all_sector_scores = []

for sector in SECTORS:
    alt_scores = np.zeros(len(ALTERNATIVES))
    for i, criterion in enumerate(CRITERIA):
        weights = st.session_state.sector_results.get((sector, criterion))
        if weights is not None:
            alt_scores += criteria_weights[i] * weights
    sector_final_scores[sector] = alt_scores
    best_index = np.argmax(alt_scores)
    sector_best_alternatives[sector] = ALTERNATIVES[best_index]
    sector_score_df = pd.DataFrame({"Alternative": ALTERNATIVES, "Score": alt_scores})
    sector_score_df["Sector"] = sector
    all_sector_scores.append(sector_score_df)

    st.success(f"✅ **{sector.title()}**: Best Alternative → **{ALTERNATIVES[best_index]}**")

st.header("④ Final AHP Between Sectoral Winners")
final_alts = [f"{SECTORS[i].title()}: {alt}" for i, alt in enumerate([sector_best_alternatives[sec] for sec in SECTORS])]
final_matrix = pairwise_matrix(final_alts, "final")
final_weights = calculate_priority_vector(final_matrix)
final_cr = consistency_ratio(final_matrix, final_weights)

final_df = pd.DataFrame({"Sector": SECTORS, "Best Alternative": [sector_best_alternatives[sec] for sec in SECTORS], "Weight": final_weights})
st.dataframe(final_df)
st.markdown(f"**Final Consistency Ratio (CR):** `{final_cr:.3f}`")
if final_cr > 0.1:
    st.warning("⚠️ High inconsistency in final AHP step.")

# --- Final Result Summary ---
st.header("🏆 Best of the Best")
best_final_index = np.argmax(final_weights)
worst_final_index = np.argmin(final_weights)

st.success(f"🏅 Best Overall Alternative: **{final_alts[best_final_index]}** with score `{final_weights[best_final_index]:.4f}`")
st.info(f"🔻 Lowest Ranked Alternative: **{final_alts[worst_final_index]}** with score `{final_weights[worst_final_index]:.4f}`")

st.header("⑤ Download Evaluation Report")
combined_df = pd.concat(all_sector_scores, ignore_index=True)
final_df["Final Weight"] = final_weights

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    pd.DataFrame({"Criteria": CRITERIA, "Weight": criteria_weights}).to_excel(writer, sheet_name='Criteria Weights', index=False)
    combined_df.to_excel(writer, sheet_name='Sector Scores', index=False)
    final_df.to_excel(writer, sheet_name='Final Evaluation', index=False)

st.download_button(
    label="📥 Download Excel Report",
    data=buffer,
    file_name="AHP_MultiSector_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
