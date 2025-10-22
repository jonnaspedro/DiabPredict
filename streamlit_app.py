import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

path = "data/diabetes.csv"
df = pd.read_csv(path)
df_output = df["Outcome"]
df_rest = df.drop(columns=["Outcome"])

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
    input_data = pd.DataFrame(
        [
            [
                pregnancies, glucose, 
                blood_pressure, skin_thickness, 
                insulin, bmi, 
                dpf, age
            ]
        ], columns=df_rest.columns
    )
    
    # --- Integração com modelo real ---
    # model = pickle.load(open("modelo_diabetes.pkl", "rb"))
    # prediction = model.predict(input_data)[0]
    
    class Net(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x
        
        
    folder = Path("model")
    files = sorted([f for f in folder.iterdir() if f.suffix == '.pt'], key=lambda f: f.name, reverse=True)
    best_model = files[0].name if len(files) > 1 else None
    if best_model is None:
        raise FileNotFoundError("Nenhum modelo encontrado em 'model/'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("model/" + best_model, map_location=device, weights_only=False)
    scaler = checkpoint["scaler"]
    
    # Essa parte leva em consideração os valores em 'model/train_model.py'.
    # Para alterar aqui você deve alterar lá e vice-versa.
    input_size = 8
    hidden_size = 32
    output_size = 2
    
    model = Net(input_size, hidden_size, output_size)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    x_new = input_data.values
    x_new_scaled = scaler.transform(x_new)
    x_new_tensor = torch.tensor(x_new_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        pred = model(x_new_tensor)
        pred_labels = pred.argmax(dim=1)
        prob = F.softmax(pred, dim=1)
    
    
    st.write("### Resultado da Previsão:")
    col1, _ = st.columns(2)

    with col1:
        if pred_labels.item() == 1:
            st.error("🩺 **Risco de Diabetes Detectado**")
            st.metric(label="Probabilidade de Diabetes", value=f"{prob[0][1].item():.1%}")
        else:
            st.success("✅ **Sem Indicação de Diabetes**")
            st.metric(label="Probabilidade de Diabetes", value=f"{prob[0][1].item():.1%}")
        
    st.warning(f"""
        ⚠️ __**Atenção!**__ ⚠️

        Inteligências Artificiais não são 100% precisas. Este resultado é apenas uma estimativa baseada em dados estatísticos.

        **DiabPredict tem aproximadamente {best_model[13:15]}% de precisão.**

        Sempre consulte um médico para diagnóstico definitivo!
    """)
