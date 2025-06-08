import streamlit as st
import numpy as np
import pandas as pd
import io

# --- Global Constants ---
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

# --- Shared Functions (AHP Methods) ---
def pairwise_matrix(items, session_key):
    n = len(items)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{session_key}_{i}_{j}"
            col1, col2, col3 = st.columns([4, 6, 4])
            with st.container():
                st.markdown(
                    "<div style='border: 2px solid #333333; border-radius: 6px; padding: 0px; margin-bottom: 5px;'>",
                    unsafe_allow_html=True,
                )
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
                    help="0 = eşit önem, pozitif = sağ daha önemli, negatif = sol daha önemli"
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

# --- TOPSIS Functions ---
def normalize_decision_matrix(matrix):
    # Her sütun için Euclidean norm ile normalizasyon
    norm_matrix = np.zeros_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        denom = np.sqrt(np.sum(col**2))
        norm_matrix[:, j] = col if denom == 0 else col / denom
    return norm_matrix

def weighted_normalized_matrix(norm_matrix, weights):
    return norm_matrix * weights

def ideal_solutions(weighted_matrix):
    positive_ideal = np.max(weighted_matrix, axis=0)
    negative_ideal = np.min(weighted_matrix, axis=0)
    return positive_ideal, negative_ideal

def calculate_topsis_scores(decision_matrix, criteria_weights):
    norm_matrix = normalize_decision_matrix(decision_matrix)
    weighted_matrix = weighted_normalized_matrix(norm_matrix, criteria_weights)
    positive_ideal, negative_ideal = ideal_solutions(weighted_matrix)
    dist_positive = np.sqrt(np.sum((weighted_matrix - positive_ideal)**2, axis=1))
    dist_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal)**2, axis=1))
    scores = dist_negative / (dist_positive + dist_negative)
    return scores

# --- Page: AHP (Orijinal Kodunuz) ---
def load_ahp_page():
    st.title("🌍 AHP Multi-Sector Decarbonization Evaluator")
    st.markdown("""
    <div style='text-align: center;'>
        Çok sektörlü decarbonizasyon stratejilerini değerlendirmek için AHP uygulamasına hoş geldiniz. 
        Aşağıdaki adımları izleyerek karar çerçevenizi oluşturabilirsiniz.
    </div>
    """, unsafe_allow_html=True)
    
    # ① Kriterlerin Karşılaştırılması
    st.header("① Kriterlerin Karşılaştırılması")
    with st.expander("Kriterleri Karşılaştır - Açmak için tıklayın"):
        st.write("Altı değerlendirme kriteri için çift yönlü karşılaştırma yapınız:")
        criteria_matrix = pairwise_matrix(CRITERIA, "criteria")
        criteria_weights = calculate_priority_vector(criteria_matrix)
        criteria_cr = consistency_ratio(criteria_matrix, criteria_weights)
        st.subheader("Kriter Ağırlıkları")
        st.dataframe(pd.DataFrame({"Kriter": CRITERIA, "Ağırlık": criteria_weights}))
        st.markdown(f"**Tutarlılık Oranı (CR):** `{criteria_cr:.3f}`")
        if criteria_cr > 0.1:
            st.warning("⚠️ Tutarsız karşılaştırma. Lütfen değerleri gözden geçirin.")
    
    # ② Sektör Bazında Alternatiflerin Değerlendirilmesi
    st.header("② Sektör Bazında Alternatiflerin Değerlendirilmesi")
    st.markdown("<div style='text-align:center;'>Tüm sektörler için alternatifleri değerlendirin:</div>", unsafe_allow_html=True)
    
    if 'sector_results' not in st.session_state:
        st.session_state.sector_results = {}
    
    sector_best_alternatives = {}
    all_sector_scores = []
    
    for sector in SECTORS:
        st.subheader(f"{sector.title()}")
        for criterion in CRITERIA:
            with st.expander(f"{criterion}", expanded=False):
                matrix = pairwise_matrix(ALTERNATIVES, f"{sector}_{criterion}")
                weights = calculate_priority_vector(matrix)
                cr = consistency_ratio(matrix, weights)
                df = pd.DataFrame({"Alternatif": ALTERNATIVES, "Ağırlık": weights})
                st.dataframe(df)
                st.markdown(f"**Tutarlılık Oranı (CR):** `{cr:.3f}`")
                if cr > 0.1:
                    st.warning("⚠️ Tutarsız karşılaştırma. Lütfen değerleri gözden geçirin.")
                st.session_state.sector_results[(sector, criterion)] = weights
    
    # ③ Her Sektör için En İyi Alternatifin Belirlenmesi
    st.header("③ Sektör Bazında En İyi Alternatifin Belirlenmesi")
    sector_final_scores = {}
    for sector in SECTORS:
        alt_scores = np.zeros(len(ALTERNATIVES))
        for i, criterion in enumerate(CRITERIA):
            weights = st.session_state.sector_results.get((sector, criterion))
            if weights is not None:
                alt_scores += criteria_weights[i] * weights
        sector_final_scores[sector] = alt_scores
        best_index = np.argmax(alt_scores)
        sector_best_alternatives[sector] = ALTERNATIVES[best_index]
        sector_score_df = pd.DataFrame({"Alternatif": ALTERNATIVES, "Skor": alt_scores})
        sector_score_df["Sektör"] = sector
        all_sector_scores.append(sector_score_df)
        st.success(f"✅ **{sector.title()}**: En İyi Alternatif → **{ALTERNATIVES[best_index]}**")
    
    # ④ Sektör Kazananları Arasında Son AHP Değerlendirmesi
    st.header("④ Sektör Kazananları Arasında Son AHP Değerlendirmesi")
    final_alts = [f"{SECTORS[i].title()}: {alt}" for i, alt in enumerate([sector_best_alternatives[sec] for sec in SECTORS])]
    final_matrix = pairwise_matrix(final_alts, "final")
    final_weights = calculate_priority_vector(final_matrix)
    final_cr = consistency_ratio(final_matrix, final_weights)
    final_df = pd.DataFrame({
        "Sektör": SECTORS,
        "En İyi Alternatif": [sector_best_alternatives[sec] for sec in SECTORS],
        "Ağırlık": final_weights
    })
    st.dataframe(final_df)
    st.markdown(f"**Son Tutarlılık Oranı (CR):** `{final_cr:.3f}`")
    if final_cr > 0.1:
        st.warning("⚠️ Final aşamasında yüksek tutarsızlık mevcut.")
    
    # ⑤ Sonuçların Özetlenmesi ve Rapor İndirimi
    st.header("🏆 En İyi Alternatif")
    best_final_index = np.argmax(final_weights)
    worst_final_index = np.argmin(final_weights)
    st.success(f"🏅 En İyi Alternatif: **{final_alts[best_final_index]}** (Skor: `{final_weights[best_final_index]:.4f}`)")
    st.info(f"🔻 En Düşük Skorlu Alternatif: **{final_alts[worst_final_index]}** (Skor: `{final_weights[worst_final_index]:.4f}`)")
    
    st.header("⑤ Değerlendirme Raporunu İndir")
    combined_df = pd.concat(all_sector_scores, ignore_index=True)
    final_df["Son Ağırlık"] = final_weights
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        pd.DataFrame({"Kriter": CRITERIA, "Ağırlık": criteria_weights}).to_excel(writer, sheet_name='Kriter Ağırlıkları', index=False)
        combined_df.to_excel(writer, sheet_name='Sektör Skorları', index=False)
        final_df.to_excel(writer, sheet_name='Final Değerlendirme', index=False)
    
    st.download_button(
        label="📥 Excel Raporunu İndir",
        data=buffer,
        file_name="AHP_MultiSector_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Ana sayfaya dönüş butonu
    if st.button("Ana Sayfaya Dön"):
        st.session_state['page'] = 'main'
        st.experimental_rerun()

# --- Page: TOPSIS (Dönüştürülmüş Kod) ---
def load_topsis_page():
    st.title("🌍 TOPSIS Multi-Sector Decarbonization Evaluator")
    st.markdown("""
    <div style='text-align: center;'>
        TOPSIS uygulamasına hoş geldiniz. 
        Bu yöntemde, alternatifler sektör bazında 1-10 arası performans puanları ile değerlendirilmekte ve TOPSIS ile sıralanmaktadır.
    </div>
    """, unsafe_allow_html=True)
    
    # ① Kriter Ağırlıkları (AHP ile)
    st.header("① Kriterlerin Karşılaştırılması (AHP)")
    with st.expander("Kriterleri Karşılaştır - Açmak için tıklayın"):
        st.write("Altı değerlendirme kriteri için çift yönlü karşılaştırma yapınız:")
        criteria_matrix = pairwise_matrix(CRITERIA, "criteria")
        criteria_weights = calculate_priority_vector(criteria_matrix)
        criteria_cr = consistency_ratio(criteria_matrix, criteria_weights)
        st.subheader("Kriter Ağırlıkları")
        st.dataframe(pd.DataFrame({"Kriter": CRITERIA, "Ağırlık": criteria_weights}))
        st.markdown(f"**Tutarlılık Oranı (CR):** `{criteria_cr:.3f}`")
        if criteria_cr > 0.1:
            st.warning("⚠️ Tutarsız karşılaştırma. Lütfen değerleri gözden geçirin.")
    
    # ② Sektör Bazında Alternatiflerin Değerlendirilmesi (TOPSIS)
    st.header("② Sektör Bazında Alternatiflerin Değerlendirilmesi (TOPSIS)")
    st.markdown("<div style='text-align:center;'>Tüm sektörler için alternatifleri değerlendirin:</div>", unsafe_allow_html=True)
    
    if 'sector_scores' not in st.session_state:
        st.session_state.sector_scores = {}
    
    for sector in SECTORS:
        st.subheader(f"{sector.title()}")
        decision_matrix = np.zeros((len(ALTERNATIVES), len(CRITERIA)))
        for idx, criterion in enumerate(CRITERIA):
            with st.expander(f"{criterion}", expanded=False):
                st.write("Her alternatif için 1-10 arası performans puanı verin:")
                scores = []
                for alt in ALTERNATIVES:
                    score = st.slider(f"{alt}", min_value=1, max_value=10, value=5, key=f"{sector}_{criterion}_{alt}")
                    scores.append(score)
                decision_matrix[:, idx] = scores
        st.session_state.sector_scores[sector] = decision_matrix
    
    # ③ Her Sektör için TOPSIS Sonuçları
    st.header("③ Sektör Bazında En İyi Alternatifin Belirlenmesi (TOPSIS Sonuçları)")
    sector_best_alternatives = {}
    all_sector_topsis_scores = []
    
    for sector in SECTORS:
        decision_matrix = st.session_state.sector_scores.get(sector)
        topsis_scores = calculate_topsis_scores(decision_matrix, criteria_weights)
        best_index = np.argmax(topsis_scores)
        sector_best_alternatives[sector] = ALTERNATIVES[best_index]
        df = pd.DataFrame({"Alternatif": ALTERNATIVES, "TOPSIS Skoru": topsis_scores})
        df["Sektör"] = sector
        all_sector_topsis_scores.append(df)
        st.success(f"✅ **{sector.title()}**: En İyi Alternatif → **{ALTERNATIVES[best_index]}**")
        st.dataframe(df)
    
    # ④ Sektör Kazananları Arasında Final TOPSIS Değerlendirmesi
    st.header("④ Sektör Kazananları Arasında Final TOPSIS Değerlendirmesi")
    final_alts = [f"{sector.title()}: {sector_best_alternatives[sector]}" for sector in SECTORS]
    st.markdown("Final alternatifleri için iki son kriter üzerinden (Overall Effectiveness, Feasibility) puan verin:")
    final_criteria = ["Overall Effectiveness", "Feasibility"]
    final_decision_matrix = np.zeros((len(final_alts), len(final_criteria)))
    for i, alt in enumerate(final_alts):
        st.markdown(f"**{alt}**")
        for j, crit in enumerate(final_criteria):
            score = st.slider(f"{crit}", min_value=1, max_value=10, value=5, key=f"final_{i}_{crit}")
            final_decision_matrix[i, j] = score
    final_weights = np.array([0.5, 0.5])
    final_topsis_scores = calculate_topsis_scores(final_decision_matrix, final_weights)
    final_results_df = pd.DataFrame({
        "Sektör Kazananı": final_alts,
        "Final TOPSIS Skoru": final_topsis_scores
    })
    st.dataframe(final_results_df)
    best_final_index = np.argmax(final_topsis_scores)
    worst_final_index = np.argmin(final_topsis_scores)
    st.success(f"🏅 En İyi Alternatif: **{final_alts[best_final_index]}** (Skor: `{final_topsis_scores[best_final_index]:.4f}`)")
    st.info(f"🔻 En Düşük Skorlu Alternatif: **{final_alts[worst_final_index]}** (Skor: `{final_topsis_scores[worst_final_index]:.4f}`)")
    
    # ⑤ Rapor İndirimi
    st.header("⑤ Değerlendirme Raporunu İndir")
    combined_df = pd.concat(all_sector_topsis_scores, ignore_index=True)
    final_results_df["Final TOPSIS Skoru"] = final_topsis_scores
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        pd.DataFrame({"Kriter": CRITERIA, "Ağırlık": criteria_weights}).to_excel(writer, sheet_name='Kriter Ağırlıkları', index=False)
        combined_df.to_excel(writer, sheet_name='Sektör TOPSIS Skorları', index=False)
        final_results_df.to_excel(writer, sheet_name='Final Değerlendirme', index=False)
    
    st.download_button(
        label="📥 Excel Raporunu İndir",
        data=buffer,
        file_name="TOPSIS_MultiSector_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Ana sayfaya dönüş butonu
    if st.button("Ana Sayfaya Dön"):
        st.session_state['page'] = 'main'
        st.experimental_rerun()

# --- Main Navigation Page ---
def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'main'
    
    if st.session_state['page'] == 'main':
        st.title("Decarbonization Evaluator")
        st.markdown("**Hangi yöntemi kullanmak istersiniz?**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("AHP"):
                st.session_state['page'] = 'ahp'
                st.experimental_rerun()
        with col2:
            if st.button("TOPSIS"):
                st.session_state['page'] = 'topsis'
                st.experimental_rerun()
    
    elif st.session_state['page'] == 'ahp':
        load_ahp_page()
    
    elif st.session_state['page'] == 'topsis':
        load_topsis_page()

if __name__ == '__main__':
    main()
