set -e

cd aegis-server
rm -rf build/*

# mkdir build/language_server

# cp -r language_server/* build/language_server


uv pip install . --target build
echo "from aegis_server.__main__ import main; main()" > build/__main__.py

uv run -m zipapp build -o ../aegis-vscode/language_server.pyz
cp ../aegis-vscode/language_server.pyz ../aegis-jetbrains/src/main/resources/language_server.pyz