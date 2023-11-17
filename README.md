```shell
conda create --name dt-test python=3.9
conda activate dt-test
git clone https://github.com/AI-secure/DecodingTrust.git && cd DecodingTrust
git checkout release
pip install -e ".[gptq]"

cd ..
git clone https://github.com/danielz02/helm.git && cd helm
pip install -e ".[gptq]"
```
