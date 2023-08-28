sphinx-apidoc -of docs dsipts/
sphinx-apidoc -of docs bash_examples/

cd docs
make clean html
cd ..
rm -rf dsipts/data_management/__pycache__
rm -rf dsipts/models/__pycache__
rm -rf dsipts/data_structure/__pycache__
rm -rf dsipts/__pycache__
rm -rf bash_examples/__pycache__

pandoc README.md -o docs/dsipts.pdf -V geometry:landscape 
cd bash_examples
pandoc README.md -o ../docs/bash_examples.pdf -V geometry:landscape 
cd ..