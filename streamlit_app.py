import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
df = pd.read_csv(url, names=columns)

st.title("DiabPredict — IA para Predição Instantânea de Diabetes")

# Visualização de Dados
st.subheader("Visualização do Dataset")
st.dataframe(df.head(10))

st.subheader("Estatísticas Descritivas")
st.write(df.describe())

#Gráficos
st.subheader("Visualização Gráfica")

# Histograma interativo
var = st.selectbox("Escolha uma variável para o histograma", df.columns[:-1])
plt.figure(figsize=(8,5))
sns.histplot(df[var], bins=20, kde=True, color='skyblue')
plt.title(f'Distribuição de {var}')
plt.xlabel(var)
plt.ylabel('Contagem')
st.pyplot(plt.gcf())
plt.clf()

# Boxplot interativo
var_box = st.selectbox("Escolha uma variável para o boxplot por diagnóstico", df.columns[:-1], index=1)
plt.figure(figsize=(8,5))
sns.boxplot(x='Outcome', y=var_box, data=df)
plt.title(f'{var_box} por Diagnóstico de Diabetes')
plt.xlabel('Diabetes (0 = Não, 1 = Sim)')
plt.ylabel(var_box)
st.pyplot(plt.gcf())
plt.clf()

# Matriz de correlação
st.subheader("Matriz de Correlação")
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(plt.gcf())
plt.clf()

# Gráfico 3D
st.subheader("Gráfico 3D: Idade x IMC x Glicose")
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
colors = df['Outcome'].map({0:'green', 1:'red'})
ax.scatter(df['Age'], df['BMI'], df['Glucose'], c=colors, s=50, alpha=0.6)
ax.set_xlabel('Idade')
ax.set_ylabel('IMC')
ax.set_zlabel('Glicose')
ax.set_title('Diabetes: Idade x IMC x Glicose')
st.pyplot(fig)
plt.clf()

# Área de IA 
st.subheader("Previsão de Diabetes com IA")
st.write("Insira os dados do paciente para prever a probabilidade de diabetes:")

pregnancies = st.number_input("Número de gestações", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glicose", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Pressão Arterial", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Espessura da Pele", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulina", min_value=0, max_value=900, value=79)
bmi = st.number_input("IMC", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Idade", min_value=0, max_value=120, value=33)

if st.button("Prever Diabetes"):
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                              columns=columns[:-1])
    
    # --- Integração com modelo real ---
    # model = pickle.load(open("modelo_diabetes.pkl", "rb"))
    # prediction = model.predict(input_data)[0]

    # Previsão de exemplo(Temos que substituir pelos dados da nossa IA)
    prediction = 1 if glucose > 125 or bmi > 30 else 0
    
    st.write("### Resultado da Previsão:")
    if prediction == 1:
        st.error("O paciente tem risco de **diabetes**.")
    else:
        st.success("O paciente tem baixo risco de diabetes.")