sphinx-apidoc -o docs dsipts/
sphinx-apidoc -o docs bash_examples/

cd docs
make clean html
cd ..
rm -rf dsipts/data_management/__pycache__
rm -rf dsipts/models/__pycache__
rm -rf dsipts/data_structure/__pycache__
rm -rf dsipts/__pycache__
rm -rf bash_examples/__pycache__