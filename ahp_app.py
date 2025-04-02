import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_weights(matrix):
    col_sum = np.sum(matrix, axis=0)
    norm_matrix = matrix / col_sum
    return np.mean(norm_matrix, axis=1)

def pairwise_input(items, session_key):
    n = len(items)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{session_key}_{i}_{j}"
            val = st.slider(f"'{items[i]}' mı yoksa '{items[j]}' mı daha önemli?", 1/9.0, 9.0, 1.0, step=0.1, key=key)
            matrix[i, j] = val
            matrix[j, i] = 1 / val
    return matrix

st.title("AHP Karar Destek Uygulaması")
st.write("Ön tanımlı kriterler ve alternatiflerle AHP hesaplaması yapın.")

# Sabit kriterler ve alternatifler
criteria_names = ['İlgi', 'Zorluk', 'Kariyer']
alt_names = ['Makine Öğrenmesi', 'Veri Tabanları', 'Optimizasyon']

st.header("1. Kriterleri Karşılaştır")
criteria_matrix = pairwise_input(criteria_names, "crit")
criteria_weights = get_weights(criteria_matrix)

alt_weights_list = []
for idx, crit in enumerate(criteria_names):
    st.header(f"2.{idx+1} - '{crit}' kriterine göre alternatifleri karşılaştır")
    alt_matrix = pairwise_input(alt_names, f"alt_{idx}")
    alt_weights = get_weights(alt_matrix)
    alt_weights_list.append(alt_weights)

# Skorları hesapla
alt_weights_list = np.array(alt_weights_list)
final_scores = np.dot(criteria_weights, alt_weights_list)

# Sonuçlar
st.header("Sonuçlar")
df = pd.DataFrame({"Alternatif": alt_names, "Skor": final_scores})
df = df.sort_values(by="Skor", ascending=False).reset_index(drop=True)
st.dataframe(df)

fig, ax = plt.subplots()
ax.bar(df["Alternatif"], df["Skor"])
ax.set_ylabel("Ağırlıklı Skor")
ax.set_title("Alternatif Karşılaştırması")
st.pyplot(fig)
