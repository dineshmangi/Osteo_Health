import streamlit as st
from streamlit_option_menu import option_menu  # Pastikan sudah install: pip install streamlit-option-menu
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Untuk memuat model

# Memuat model yang telah dilatih
random_forest_model = joblib.load('random_forest_model.sav')  # Ganti dengan path model Anda
svm_model = joblib.load('svm_model.sav')  # Ganti dengan path model Anda
gradient_boosting_model = joblib.load('gradient_boosting_model.sav')  # Ganti dengan path model Anda

# Membuat menu samping
with st.sidebar:
    selected = option_menu(
        "Menu", ["Home", "Prediksi", "Visualisasi"],
        icons=['house', 'activity', 'bar-chart-line'],
        menu_icon="menu-button", default_index=0)

# Halaman Home
if selected == "Home":
    st.title("Aplikasi Prediksi Penyakit Osteoporosis")
    st.write("**Oleh : Rajendra Artanto - 21.11.4236**")

    #st.image('D:\KULIAH\Osteoporosis\Persiapan Deploy\IMG\osteo_head.jpg')
    st.image('img/osteo_head.jpg', use_container_width=True)
    st.write("""
             
    Aplikasi ini bertujuan untuk membantu tenaga medis dalam mendiagnosis dan memprediksi penyakit Osteoporosis. 
    Dengan menggunakan teknologi kecerdasan buatan dan analisis data, aplikasi ini mampu memberikan prediksi yang 
    akurat berdasarkan data medis pasien, seperti riwayat kesehatan, dan faktor risiko lainnya. 
    Dengan aplikasi ini, tenaga medis dapat meningkatkan akurasi diagnosis, mempercepat proses pengambilan keputusan, 
    dan memberikan perawatan yang lebih efektif kepada pasien yang berisiko Osteoporosis.
    """)

    st.subheader("Tentang Algoritma Prediksi")
    with st.expander("Random Forest"):
        st.image('img/srandomforest.png', use_container_width=True)
        #st.image('D:\KULIAH\Osteoporosis\Persiapan Deploy\IMG\srandomforest.png')
        st.write("Random forest adalah algoritma yang menggabungkan hasil (output) dari beberapa decision tree untuk mencapai satu hasil yang lebih akurat. Random forest membutuhkan gabungan beberapa decision tree untuk memprediksi hasil yang akurat. ")
        st.write("Konsep sederhana dari random forest adalah beberapa decision tree yang tidak berkorelasi akan bekerja lebih baik sebagai kelompok dibandingkan individu. Saat menggunakan random forest sebagai pengklasifikasi, satu decision tree menyumbang satu suara. Setiap decision tree bisa menghasilkan jawaban yang sama atau berbeda satu sama lain. ")
        st.write("Misalnya decision tree A, B, E dan F memprediksi hasil 1. Sementara decision tree C dan D memprediksi hasil 0. Karena ada banyaknya alternatif jawaban dalam decision tree dan kemungkinan bias yang tinggi, random forest mengambil prediksi hasil dari beberapa decision tree berdasarkan suara mayoritas dan memprediksi hasil yang lebih akurat.")
        st.write("Semakin banyak hasil decision tree yang diambil, semakin tinggi akurasi terutama ketika masing-masing pohon tidak berkorelasi satu sama lain.  ")

    with st.expander("Support Vector Machine"):
        st.image('img/svm.jpg',use_container_width=True)
        #st.image('D:\KULIAH\Osteoporosis\Persiapan Deploy\IMG\svm.jpg')
        st.write("Support Vector Machine (SVM) adalah algoritma machine learning yang digunakan untuk tugas klasifikasi dan regresi, namun paling dikenal dalam klasifikasi. SVM bekerja dengan mencari hyperplane terbaik yang memisahkan dua kelas dalam dataset dengan margin yang maksimal. Titik-titik data yang paling dekat dengan hyperplane disebut support vectors, yang menentukan posisi hyperplane tersebut.")
        st.write("SVM berfokus pada memaksimalkan margin, yaitu jarak antara hyperplane dengan support vectors dari masing-masing kelas, untuk menghasilkan model yang lebih generalis dalam menghadapi data baru. Jika data tidak dapat dipisahkan secara linier, SVM menggunakan metode kernel trick untuk memetakan data ke ruang berdimensi lebih tinggi sehingga dapat dipisahkan. Algoritma ini sangat efektif dalam ruang dimensi tinggi dan mampu menangani masalah klasifikasi non-linier dengan baik. ")

    with st.expander("Gradient Boosting"):
        st.image('img/gradientboosting.png',use_container_width=True)
        #st.image('D:\KULIAH\Osteoporosis\Persiapan Deploy\IMG\gradientboosting.png')
        st.write("Gradient Boosting adalah salah satu metode Machine Learning yang berfokus pada perbaikan kinerja model melalui peningkatan performa model sebelumnya. Algoritma ini menggunakan pendekatan boosting yang melibatkan peningkatan performa model dengan memanfaatkan informasi dari model-model sebelumnya.")
        st.write("Gradient Boosting merupakan algoritma machine learning yang menggabungkan beberapa model kecil menjadi satu model yang lebih kuat dan lebih baik dalam memprediksi data. Algoritma ini bekerja dengan mengukur eror dari model sebelumnya dan menggunakan informasi tersebut untuk memperbaiki performa model berikutnya.")
        st.write("Gradient Boosting terkenal akan kemampuannya untuk menangani data yang kompleks dan memberikan prediksi yang akurat dengan menggabungkan kekuatan dari beberapa model lemah untuk membentuk model yang kuat. Teknik ini juga fleksibel dan dapat digunakan untuk berbagai jenis data, baik regresi maupun klasifikasi.")
        st.write("Namun, kelemahan Gradient Boosting sendiri adalah bahwa proses trainingnya cenderung lambat dan membutuhkan sumber daya komputasi yang tinggi, terutama pada dataset besar. Selain itu, model ini rentan terhadap overfitting jika tidak dikonfigurasi dengan baik, memerlukan pemilihan parameter yang hati-hati dan validasi yang tepat untuk mencapai kinerja optimal.")


# Halaman Prediksi
elif selected == "Prediksi":
    st.title("Prediksi Penyakit Osteoporosis")

    # Membuat tiga kolom untuk input data
    col1, col2, col3 = st.columns(3)

    # Input pengguna pada kolom 1
    with col1:
        Age = st.number_input("Usia", min_value=0, max_value=120, value=18)
        Gender = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
        Body_Weight = st.selectbox("Berat Badan", ["Normal", "Kurus"])
        Hormonal_Changes = st.selectbox("Perubahan Hormonal", ["Normal", "Pasca-menopause"])
        Medications = st.selectbox("Obat yang Digunakan", ["Kortikosteroid", "Tidak ada"])

    # Input pengguna pada kolom 2
    with col2:
        Family_History = st.selectbox("Riwayat Keluarga", ["Ya", "Tidak"])
        Race_Ethnicity = st.selectbox("Ras/Etnis", ["Kaukasia", "Afrika-Amerika", "Asia"])
        Calcium_Intake = st.selectbox("Asupan Kalsium", ["Tercukupi", "Kurang"])
        Vitamin_D_Intake = st.selectbox("Asupan Vitamin D", ["Tercukupi", "Kurang"])
        Prior_Fractures = st.selectbox("Fraktur Sebelumnya", ["Ya", "Tidak"])

    # Input pengguna pada kolom 3
    with col3:
        Physical_Activity = st.selectbox("Aktivitas Fisik", ["Aktif", "Kurang Gerak"])
        Smoking = st.selectbox("Merokok", ["Ya", "Tidak"])
        Alcohol_Consumption = st.selectbox("Konsumsi Alkohol", ["Ya", "Tidak"])
        Medical_Conditions = st.selectbox("Kondisi Medis", ["Artritis Reumatoid", "Hipertiroidisme", "Tidak ada"])

    # Pilihan model
    model_choice = st.selectbox("Pilih Model", ["Random Forest", "SVM", "Gradient Boosting"])

    # Tombol prediksi
    if st.button('Prediksi'):
        # Konversi input menjadi format yang benar
        Gender = 1 if Gender == 'Pria' else 0
        Body_Weight = 1 if Body_Weight == 'Kurus' else 0
        Hormonal_Changes = 1 if Hormonal_Changes == 'Pasca-menopause' else 0
        Family_History = 1 if Family_History == 'Ya' else 0
        Race_Ethnicity = {'Kaukasia': 1, 'Afrika-Amerika': 2, 'Asia': 3}[Race_Ethnicity]
        Calcium_Intake = 1 if Calcium_Intake == 'Tercukupi' else 0
        Vitamin_D_Intake = 1 if Vitamin_D_Intake == 'Tercukupi' else 0
        Physical_Activity = 1 if Physical_Activity == 'Aktif' else 0
        Smoking = 1 if Smoking == 'Ya' else 0
        Alcohol_Consumption = 1 if Alcohol_Consumption == 'Ya' else 0
        Medical_Conditions = {'Artritis Reumatoid': 1, 'Hipertiroidisme': 2, 'Tidak ada': 0}[Medical_Conditions]
        Medications = 1 if Medications == 'Kortikosteroid' else 0
        Prior_Fractures = 1 if Prior_Fractures == 'Ya' else 0

        input_data = np.array([[Age, Gender, Body_Weight, Hormonal_Changes, Family_History, Race_Ethnicity,
                                Calcium_Intake, Vitamin_D_Intake, Physical_Activity, Smoking,
                                Alcohol_Consumption, Medical_Conditions, Medications, Prior_Fractures]])

        # Lakukan prediksi berdasarkan model yang dipilih
        if model_choice == "Random Forest":
            osteoporosis_diagnosis = random_forest_model.predict(input_data)
            probabilities = random_forest_model.predict_proba(input_data)
        elif model_choice == "SVM":
            osteoporosis_diagnosis = svm_model.predict(input_data)
            probabilities = svm_model.predict_proba(input_data)
        elif model_choice == "Gradient Boosting":
            osteoporosis_diagnosis = gradient_boosting_model.predict(input_data)
            probabilities = gradient_boosting_model.predict_proba(input_data)

        if osteoporosis_diagnosis[0] == 1:
            osteoporosis_diagnosis = 'Pasien Terkena penyakit Osteoporosis'
            probability = f"Probabilitas: {probabilities[0][1] * 100:.2f}%"
            st.markdown(f"<span style='color: green;'>{osteoporosis_diagnosis}</span>", unsafe_allow_html=True)
        else:
            osteoporosis_diagnosis = 'Pasien Tidak Terkena penyakit Osteoporosis'
            probability = f"Probabilitas: {probabilities[0][0] * 100:.2f}%"
            st.markdown(f"<span style='color: red;'>{osteoporosis_diagnosis}</span>", unsafe_allow_html=True)

        # Tampilkan probabilitas
        st.info(probability)

elif selected == "Visualisasi":

    df_visual = pd.read_excel('osteocross.xlsx')

    st.header("Visualisasi Cross Tabulation Masing-Masing Fitur dengan Diagram Batang")
    st.write("Visualisasi menggunakan dataset yang telah melalui tahap pra-pemrosesan data yang meliputi handling null values menggunakan teknik imputasi data ")

    # List semua fitur kecuali kolom age
    columns_to_visualize = df_visual.columns[df_visual.columns != 'Age']


    col1, col2 = st.columns(2)
    # Looping untuk membuat crosstab dan visualisasi
    with col1:
        st.subheader("Visualisasi cross-tabulation untuk pasien kategori 1 (Osteoporosis):")

        for feature in columns_to_visualize:
            st.write(f"### Kolom: {feature}")
        
            # Crosstab antara fitur dan target (misalnya 'osteoporosis' sebagai label target)
            crosstab = pd.crosstab(df_visual[feature], df_visual['Osteoporosis'])

            # Menghitung data untuk kategori Osteoporosis (kategori 1) saja
            osteoporosis_data = df_visual[df_visual['Osteoporosis'] == 1]
            count_by_feature = osteoporosis_data[feature].value_counts()

            # Plot crosstab menggunakan seaborn barplot
            plt.figure(figsize=(8, 6))
            colors = ['#87CEEB', '#FFA500', '#008000']  # Warna untuk bar (sesuaikan jika fitur berbeda)
            ax = sns.barplot(x=count_by_feature.index, y=count_by_feature.values, palette=colors, width=0.5)

            # Tambahkan label di dalam bar
            for i, v in enumerate(count_by_feature.values):
                ax.text(i, v - v * 0.1, str(v), color='black', ha='center', fontweight='bold')  # Label di dalam bar

            plt.title(f'Distribusi Osteoporosis (Kategori 1) Berdasarkan {feature}')
            plt.xlabel(feature)
            plt.ylabel('Jumlah')
        
            # Tampilkan plot di Streamlit
            st.pyplot(plt)
            plt.clf()  # Membersihkan plot untuk mencegah overlay di plot berikutnya
    

    with col2:
        st.subheader("Visualisasi cross-tabulation untuk pasien kategori 0 (Non-Osteoporosis):")

        for feature in columns_to_visualize:
            st.write(f"### Kolom: {feature}")
        
            # Crosstab antara fitur dan target (misalnya 'osteoporosis' sebagai label target)
            crosstab = pd.crosstab(df_visual[feature], df_visual['Osteoporosis'])

            # Menghitung data untuk kategori Osteoporosis (kategori 1) saja
            osteoporosis_data = df_visual[df_visual['Osteoporosis'] == 0]
            count_by_feature = osteoporosis_data[feature].value_counts()

            # Plot crosstab menggunakan seaborn barplot
            plt.figure(figsize=(8, 6))
            colors = ['#87CEEB', '#FFA500', '#008000']  # Warna untuk bar (sesuaikan jika fitur berbeda)
            ax = sns.barplot(x=count_by_feature.index, y=count_by_feature.values, palette=colors, width=0.5)

            # Tambahkan label di dalam bar
            for i, v in enumerate(count_by_feature.values):
                ax.text(i, v - v * 0.1, str(v), color='black', ha='center', fontweight='bold')  # Label di dalam bar

            plt.title(f'Distribusi Non-Osteoporosis (Kategori 0) Berdasarkan {feature}')
            plt.xlabel(feature)
            plt.ylabel('Jumlah')
        
            # Tampilkan plot di Streamlit
            st.pyplot(plt)
            plt.clf() 
