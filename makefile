install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt &&\
	python setup.py install
	
format:
	black ./src/ADMiniSter/*.py
	
lint:
	pylint --disable=R,C ./src/ADMiniSter/*.py
	
test:
	python -m pytest --cov=ADMiniSter -vv ./src/ADMiniSter/
	
all: install format lint test
