# 💉 DiabPredict — IA para Predição Instantânea de Diabetes

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-v1.30-orange) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-green) ![MIT License](https://img.shields.io/badge/License-MIT-brightgreen)

**DiabPredict** é uma solução inovadora que utiliza **Inteligência Artificial** para estimar o risco de diabetes **imediatamente após um exame de sangue**. A ferramenta combina **Machine Learning** com uma interface intuitiva, permitindo que profissionais de saúde e pacientes recebam um diagnóstico rápido e confiável.

## 🔹 Funcionalidades

- ⚡ **Predição instantânea** de diabetes a partir dos dados do exame de sangue  
- 🖥️ **Interface amigável** para inserção dos resultados  
- 📊 Modelo treinado com o **Pima Indians Diabetes Dataset**  
- ✅ Resultado claro: **“Risco de Diabetes: Sim”** ou **“Não”**

---

## 🔹 Como funciona

1. 🩺 Realize o exame de sangue  
2. 📝 Insira os resultados no sistema (glicose, pressão arterial, insulina, IMC, idade e outros parâmetros clínicos)  
3. 🤖 A IA processa os dados e retorna instantaneamente o **risco de diabetes**

---

## 🔹 Dataset

O projeto utiliza o **Pima Indians Diabetes Dataset** do UCI Machine Learning Repository:  
[📄 Link do dataset](https://archive.ics.uci.edu/dataset/34/pima+indians+diabetes)

---

## 🔹 Tecnologias e Bibliotecas

- **Python 3**  
- **Machine Learning:** scikit-learn (Regressão Logística, Random Forest, SVM)  
- **Interface:** Streamlit  
- **Análise e visualização de dados:** pandas, numpy, matplotlib, seaborn  

---

## 🔹 Como testar a IA 🚀

1️⃣ Execute o app Streamlit:  

```bash
streamlit run app/app.py
```
2️⃣ Preencha os resultados do exame de sangue:  

- Glicose  
- Pressão Arterial  
- Insulina  
- IMC  
- Idade  
- Outros parâmetros clínicos

3️⃣ Clique em **"Prever"** e visualize o resultado:  

✅ Sem risco de diabetes  
⚠️ Risco de diabetes

💡 **Dica:** Experimente diferentes valores para entender como cada parâmetro influencia a predição da IA

---

## 🔹 Autores 👨‍💻

**Jonnas Pedro**, **Cauã Rocha** e **João Farias** — desenvolvimento do projeto como parte da **Atividade AV3**

---

## 🔹 Licença 📜

Este projeto está licenciado sob a **MIT License**
