import numpy as np

def ahp_priority_vector(pairwise_mat):
    """
    Verilen pairwise comparison (ikili karşılaştırma) matrisi için
    en büyük özdeğer (principal eigenvalue) ve o özdeğere karşılık
    gelen özvektörden yola çıkarak normalleştirilmiş öncelik vektörünü döndürür.
    Aynı zamanda tutarlılık oranını (consistency ratio) da hesaplar.
    """
    # Adım 1: Özdeğer ve özvektörleri bul
    eigenvalues, eigenvectors = np.linalg.eig(pairwise_mat)
    
    # En büyük özdeğerin indeksi
    max_eig_index = np.argmax(eigenvalues.real)
    
    # İlgili özvektör
    max_eig_vector = eigenvectors[:, max_eig_index].real
    
    # Normalizasyon
    priority_vector = max_eig_vector / np.sum(max_eig_vector)
    
    # Tutarlılık oranı için gerekli hesaplamalar
    n = pairwise_mat.shape[0]
    max_eig_val = eigenvalues[max_eig_index].real
    
    # Rastgele tutarlılık indexi (RI) tablosu (Saaty 1-10 arası için)
    # n = 6 için RI = 1.24 olarak alabiliriz.
    RI_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_dict.get(n, 1.24)  # n=6 için 1.24
    
    CI = (max_eig_val - n) / (n - 1)
    CR = CI / RI if RI != 0 else 0
    
    return priority_vector, CR

# 1. Kriterler Arası Karşılaştırma Matrisi
# 6 kriterimiz var: [Sıkılmamak, Arkadaş, GelişPot, SoruKatsayısı, ŞuAnkiDurum, Zorluk]
# Buradaki örnek değerler tamamen hayalidir. Siz kendi uzman görüşünüze göre 1-9 ölçeğinde doldurun.
criteria_pairwise = np.array([
    [1,   3,   5,   1,   3,   2],  # Sıkılmamak'ın diğer kriterlere göre önemi
    [1/3, 1,   2,   1/2, 2,   1/2],
    [1/5, 1/2, 1,   1/3, 1,   1/2],
    [1,   2,   3,   1,   3,   2],
    [1/3, 1/2, 1,   1/3, 1,   1/2],
    [1/2, 2,   2,   1/2, 2,   1]
], dtype=float)

criteria_weights, criteria_CR = ahp_priority_vector(criteria_pairwise)
print("Kriterlerin Öncelik Vektörü (Ağırlıklar):", criteria_weights)
print("Kriterler Matrisinin Tutarlılık Oranı (CR):", criteria_CR)
print("--------------------------------------------------------------")

# -- Adım 2: Her ders için 5 uygulamayı kriter bazında karşılaştırma --
lessons = ["Fizik", "Matematik", "Kimya", "Biyoloji", "Tarih"]
applications = ["Deneme Çözmek", "Konu Çalışmak", "Soru Çözmek", "Eski Denemeleri Analiz Etmek", "Özel Ders Almak"]

# Örnek olarak sadece "Fizik" dersi için 6 kriterden 1 tanesi hakkında (örneğin Sıkılmamak) 5 uygulamanın pairwise matrisini gösterelim.
# Gerçekte her ders ve her kriter için 5x5 boyutunda ayrı bir matris girmek gerekiyor (yani 5 ders x 6 kriter = 30 matris).
# Bu çok uzun olduğundan burada sadece bir örnek gösteriyoruz. Siz tüm matrisleri doldurmalısınız.

# Fizik - Sıkılmamak kriterine göre 5 uygulamanın karşılaştırma matrisi (örnek değerler)
fizik_sikilma_pairwise = np.array([
    [1,   3,   2,   4,   1],  # Deneme Çözmek'in diğer uygulamalara göre önemi
    [1/3, 1,   1/2, 2,   1/3],
    [1/2, 2,   1,   3,   1/2],
    [1/4, 1/2, 1/3, 1,   1/5],
    [1,   3,   2,   5,   1]
], dtype=float)

# Bu matristen yerel ağırlıklar ve CR hesaplayalım (sadece örnek):
fizik_sikilma_weights, fizik_sikilma_CR = ahp_priority_vector(fizik_sikilma_pairwise)
print("Fizik dersi, 'Sıkılmamak' kriterine göre uygulama ağırlıkları:", fizik_sikilma_weights)
print("Tutarlılık Oranı:", fizik_sikilma_CR)
print("--------------------------------------------------------------")

# Bu şekilde:
#   - Fizik dersi için "Sıkılmamak" kriteri -> 5x5 matris
#   - Fizik dersi için "Arkadaş" kriteri -> 5x5 matris
#   - ...
#   - Fizik dersi için "Zorluk" kriteri -> 5x5 matris
# Tümünün yerel öncelik vektörlerini bulacağız.
# Sonra bu yerel öncelikleri, yukarıda bulduğumuz "criteria_weights" (6 kriterin ağırlıkları) ile çarparak
# 5 uygulamanın (Deneme, Konu, Soru, Eski Deneme Analizi, Özel Ders) nihai puanlarını elde edeceğiz.

# Örnek olarak sadece 1 kriter üzerinden nihai skor hesaplaması yapalım:
# diyelim ki "Sıkılmamak" kriterinin global ağırlığı:
sikilma_index = 0  # Kriterler listesinde 0. sırada olduğunu varsayıdık
sikilma_global_weight = criteria_weights[sikilma_index]

# O halde Fizik - Sıkılmamak kriteri için uygulama skorları:
fizik_sikilma_global_scores = fizik_sikilma_weights * sikilma_global_weight
print("Fizik (Sıkılmamak kriteri) Uygulama Skorları (Global):")
print(fizik_sikilma_global_scores)
print("--------------------------------------------------------------")


# Siz bunu her bir kriter için yapacak (toplam 6 kriter) ve ardından
# toplama işlemiyle (veya çarpımla değil, AHP toplama yaklaşımı) her bir uygulamanın
# tüm kriterler altındaki toplam global skorunu elde edeceksiniz.

# Örneğin (tamamı hayali rakam):
#   Fizik_Uygulama1_toplam_skor = sum( [ Fizik_Sıkılmamak_local[i]*KriterAğırlık[i] + ... ] )
#   Fizik_Uygulama2_toplam_skor = ...
#   ...
#   En yüksek skor -> o dersin seçilecek uygulaması.

# Bunu tüm dersler için yaptıktan sonra elinizde şöyle bir sonuç listesi olsun (örnek):
best_app_for_lesson = {
    "Fizik": "Deneme Çözmek",
    "Matematik": "Soru Çözmek",
    "Kimya": "Konu Çalışmak",
    "Biyoloji": "Deneme Çözmek",
    "Tarih": "Özel Ders Almak"
}

# -- Adım 3: Ders-Uygulama çiftlerini yeni bir matrise alıp tekrar AHP uygulama --

# Şimdi her dersin en iyi uygulaması belli oldu. 5 tane (ders-uygulama) çiftimiz var.
lesson_app_pairs = list(best_app_for_lesson.items())  # [("Fizik","Deneme Çözmek"), ("Matematik","Soru Çözmek"), ...]

print("Ders - Uygulama Çiftleri:")
for pair in lesson_app_pairs:
    print(pair)
print("--------------------------------------------------------------")

# Bu 5 çiftin her birini bir 'alternatif' olarak düşünün.
# Tekrar 6 kriter üzerinden (veya isterseniz kriter öncelik matrisini güncelleyerek) 
# bir 5x5 pairwise comparison matrisi oluşturup, hangi (Ders-Uygulama) çiftini 
# ilk olarak yapmanız gerektiğini AHP ile belirleyebilirsiniz.

# Örneğin ders-uygulama çiftlerini Sıkılmamak kriteri açısından 5x5 pairwise matrisi (örnek değer):
final_pairwise_sikilma = np.array([
    [1,   3,   2,   4,   2],   # (Fizik,Deneme) vs diğerleri
    [1/3, 1,   2,   3,   2],
    [1/2, 1/2, 1,   2,   1],
    [1/4, 1/3, 1/2, 1,   1/2],
    [1/2, 1/2, 1,   2,   1]
], dtype=float)

# ... ve bu şekilde diğer 5 kriter için de (Ders-Uygulama) çiftlerine yönelik 5x5 matrisler girilecektir.

# Tüm kriterlerdeki yerel öncelikler + kriter ağırlıkları kullanılarak,
# 5 alternatifin (Ders-Uygulama çiftinin) nihai öncelik skoru elde edilir.

# Nihai skorun en yüksek olduğu (Ders-Uygulama) çifti, AHP yaklaşımına göre
# ilk çalışılması gereken ders + uygulanacak yöntem olarak karşımıza çıkar.

# =============================
# Özetle Yapmanız Gerekenler:
# =============================
# 1) 6 kriter için 6x6 pairwise matrisi -> kriter_weights (ve CR)
# 2) Her ders (5 adet) ve her kriter (6 adet) için 5x5 boyutunda pairwise matrisi:
#    -> 5 (ders) x 6 (kriter) x (5x5) = 30 matrisi doldurun.
# 3) Her matrise göre uygulamaların yerel ağırlıkları bulun.
# 4) Bu yerel ağırlıkları, ilgili kriterin global ağırlığı (kriter_weights) ile çarparak
#    uygulamaların o derse özel toplam skorlarını elde edin. En yüksek skor -> o dersin en iyi uygulaması.
# 5) (Ders + En İyi Uygulama) olarak 5 çiftiniz olacak. Bu 5 çift, AHP’de “alternatif” gibi düşünülür.
#    Aynı 6 kriter veya yeni ek kriterlerle 5 adet alternatif için 5x5 pairwise matrisi oluşturun.
#    6 kriterin ağırlıklarıyla birleştirerek final skorlarınızı hesaplayın.
# 6) Elde ettiğiniz final skorlara göre ilk sırada yapılması gereken (Ders-Uygulama) çiftini seçin.

# Kod içinde yer alan matris değerleri tamamen örnek niteliğindedir;
# kendi uzmanlık veya tercihlerinize göre 1–9 ölçeğinde mantıklı karşılaştırma değerlerini girmeniz gerekir.