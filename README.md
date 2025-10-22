## 💉 DiabPredict — IA para Predição Instantânea de Diabetes

![Python](https://img.shields.io/badge/Python-3.12.4-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-v1.50.0-orange) ![MIT License](https://img.shields.io/badge/License-MIT-brightgreen)

**DiabPredict** é uma solução inovadora que utiliza **Inteligência Artificial** para estimar o risco de diabetes **imediatamente após um exame de sangue**. A ferramenta combina **Machine Learning** com uma interface intuitiva, permitindo que profissionais de saúde e pacientes recebam um diagnóstico rápido e confiável.

## 🔹 Funcionalidades

- ⚡ **Predição instantânea** de diabetes a partir dos dados do exame de sangue  
- 🖥️ **Interface amigável** para inserção dos resultados  
- 📊 Modelo treinado com o **Pima Indians Diabetes Dataset**  
- ✅ Resultado claro: **“Risco de Diabetes: Sim”** ou **“Não”**


## 🔹 Como funciona

1. 🩺 Realize o exame de sangue  
2. 📝 Insira os resultados no sistema (glicose, pressão arterial, insulina, IMC, idade e outros parâmetros clínicos)  
3. 🤖 A IA processa os dados e retorna instantaneamente o **risco de diabetes**


## 🔹 Dataset

O projeto utiliza o **Pima Indians Diabetes Dataset** do UCI Machine Learning Repository:  
[📄 Link do dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)


## 🔹 Tecnologias e Bibliotecas

* **[Dependências (uso apenas da CPU)](requirements-cpu)**<br>
* **[Dependências (suporte a GPU com CUDA)](requirements-gpu)**


## 🔹 Como testar a IA 🚀
1️⃣ Instale as dependências:
```bash
# Se você não quer utilizar CUDA ou não tem suporte em sua GPU, execute isso:
pip install -r requirements-cpu.txt
# Se você quer utilizar CUDA e tem suporte em sua GPU, execute isso:
pip install -r requirements-gpu.txt
```

2️⃣ Para rodar tudo junto:
```bash
# Certifique-se de baixar o Makefile antes
make run
```

**Ou, se quiser separadamente:**

2️⃣ Traine o seu modelo:
```bash
make train
# ou
python model/train_model.py
```

3️⃣ Execute o app Streamlit:  

```bash
make run
# ou
streamlit run streamlit_app.py
```

4️⃣ Preencha os resultados do exame de sangue:  

- Glicose  
- Pressão Arterial  
- Insulina  
- IMC  
- Idade  
- Outros parâmetros clínicos

5️⃣ Clique em **"Prever"** e visualize o resultado:  

✅ Sem risco de diabetes  
⚠️ Risco de diabetes

💡 **Dica:** Experimente diferentes valores para entender como cada parâmetro influencia a predição da IA

## 🔹 Artigo Científico 📖

Predição de Diabetes Utilizando Modelos de Aprendizado de Máquina com o Dataset Pima Indians

O presente trabalho apresenta o DiabPredict, uma aplicação baseada em Inteligência Artificial voltada para a predição instantânea do risco de diabetes a partir de dados clínicos de exames de sangue. classificador baseado em rede neural totalmente conectada, otimizado por algoritmo genético para prever a ocorrência de diabetes. A ferramenta foi desenvolvida em Python com interface interativa em Streamlit, permitindo fácil utilização por profissionais de saúde e pacientes. Os resultados indicam que o uso de algoritmos de aprendizado supervisionado pode auxiliar de forma eficaz na identificação precoce do diabetes, contribuindo para diagnósticos mais rápidos e decisões médicas assertivas.

📄 [Leia o artigo completo do DiabPredict](https://github.com/user-attachments/files/23045067/DiabPredict_TDS_IFPE.pdf) 
## 🔹 Autores 👨‍💻

**Jonnas Pedro**, **Cauã Rocha** e **João Vitor** — desenvolvimento do projeto como parte da **Atividade AV3**


## 🔹 Licença 📜

Este projeto está licenciado sob a **MIT License**
