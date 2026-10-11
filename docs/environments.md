# Task environments

Training uses separate `rllm` (Python 3.11.8) and `rllm-webshop` (Python 3.10.0) conda environments. Their shared inference stack and task-specific TensorDict versions are listed in the [README](../README.md#installation). Both use Ray 2.54.0 and PyTorch 2.6.0+cu124.

## Math and LiveCodeBench

Use the `rllm` environment with TensorDict 0.8.3. Preparation downloads public datasets from Hugging Face. The installed math-checking packages were SymPy 1.13.1, pylatexenc 2.10, and antlr4-python3-runtime 4.9.3. The release declares `math-verify` and `latex2sympy2_extended` as additional dependencies; neither was installed in the recorded experiment environments. Code rewards execute generated Python with resource limits; run code experiments in an isolated Linux training environment.

## AppWorld

Use a separate Python 3.11 environment for AppWorld's Pydantic-v1 dependencies. The preserved environment used AppWorld `0.1.3.post1`, FastAPI `0.110.3`, and Pydantic `1.10.26`.

```bash
python3.11 -m venv .venv-appworld
source .venv-appworld/bin/activate
pip install appworld==0.1.3.post1 fastapi==0.110.3 pydantic==1.10.26
export APPWORLD_ROOT=/path/to/appworld-data
appworld install
appworld download data
python scripts/export_appworld_task_ids.py --output /path/to/appworld-task-ids.json
```

Start each environment server in its own terminal/process:

```bash
appworld serve environment --port 8100
appworld serve environment --port 8101
```

The client allocates one world per server and queues concurrent rollouts until a server is available. For the default concurrency of eight, start eight servers on ports 8100–8107. In the training environment, set `APPWORLD_REMOTE_ENVIRONMENT_URLS` to their comma-separated URLs, and prepare the registry using `APPWORLD_TASK_IDS_FILE`. The training environment needs no AppWorld package.

See the [AppWorld installation and serving documentation](https://github.com/StonyBrookNLP/appworld/blob/main/README.md) for backend details. The package version above fixes the API used by this adapter.

## WebShop

Use the dedicated **`rllm-webshop` environment**, with Python 3.10.0 and TensorDict 0.6.2, for data preparation, training, and evaluation. Its additional installed packages were:

| Package | Version |
| --- | --- |
| OpenJDK (conda) | 11.0.23 |
| Pyserini | 0.17.0 |
| PyJNIus | 1.7.0 |
| Gym | 0.24.0 |
| spaCy | 3.8.11 |
| Flask | 2.1.2 |
| BeautifulSoup4 | 4.11.1 |
| rank-bm25 | 0.2.2 |
| FastAPI | 0.125.0 |
| Pydantic | 2.12.0 |

Activate `rllm-webshop` before setting up or running the backend:

```bash
conda activate rllm-webshop
```

The adapter imports `WebAgentTextEnv` from a local WebShop checkout and uses its product corpus and Java search index. Follow [WebShop setup](https://github.com/princeton-nlp/WebShop#-setup) to download products, install dependencies, and build the index. Keep those dependencies accessible to the training Python process.

```bash
git clone https://github.com/princeton-nlp/WebShop.git /path/to/webshop
cd /path/to/webshop
bash setup.sh -d small
```

Set `WEBSHOP_ROOT` to that directory. The experiment used OpenJDK from the WebShop conda environment. Point `JAVA_HOME` to the OpenJDK 11 installation used by this environment when explicit JVM selection is needed. The recovered backend defaults to 1,000 products and `human_goals=0`; `WEBSHOP_NUM_PRODUCTS` and `WEBSHOP_HUMAN_GOALS` configure these values. Use the same corpus settings for data preparation, training, and evaluation.

The preparation script retains the recovered goal ordering: the test prefix is `[0, test_size)`, the next 1,000 goals are validation goals, and the remaining goals form the training pool. It samples 2,400 tasks from that training pool using the configured seed. The default test prefix contains 100 tasks. Increasing `--test_size` changes the pool boundary, so regenerate the registry before changing that setting.
