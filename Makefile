PYTHON ?= python

all: train run

train:
	@echo "Treinando o modelo..."
	$(PYTHON) model/train_model.py

run:
	@echo "Iniciando o Streamlit..."
	streamlit run streamlit_app.py

clean:
	@echo "Limpando arquivos temporários..."
	rm -rf __pycache__ .streamlit-cache
