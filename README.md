# RAG-exp


1. create the virtual environment (you can also use conda to create the environment)

For CPU-only environment:

```bash
python3 -m venv ragEnv
source ragEnv/bin/activate
pip install requirements-cpu.txt 
```

For cuda environment:

```bash
python3 -m venv ragEnv
source ragEnv/bin/activate
pip install requirements-cuda.txt 
```

2. using the mistralai API:

- get the API key from [mistralai](https://console.mistral.ai/api-keys/). In the workspace billing, just select the "Experiment" option. It is free!! Update the api_key in ```run.py```

- run the RAG:

```bash
python run-mistralai.py
```

3. using open source models from HuggingFace (you can change the ```mdoel_id``` in the code to try other models):

```bash
python run_open_llm.py
```