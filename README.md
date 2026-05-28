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

** More details can be found at our publised work:

<pre><code>
@ARTICLE{11300859,
  author={Amaral Cejas, Orlando and Guo, Yuejun and Tang, Qiang},
  journal={IEEE Access}, 
  title={From Retrieval to Response: Tracing the Impact of Embedding Quality in RAG Systems}, 
  year={2025},
  volume={13},
  number={},
  pages={212773-212781},
  doi={10.1109/ACCESS.2025.3644595}}

</code></pre>


If you use this project, please consider citing us.
