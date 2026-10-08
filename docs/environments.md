# Task environments

## Math and LiveCodeBench

Preparation downloads public datasets from Hugging Face. The training environment supplies the verifier dependencies. Code rewards execute generated Python with resource limits; run code experiments in an isolated Linux training environment.

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

The adapter imports `WebAgentTextEnv` from a local WebShop checkout and uses its product corpus and Java search index. Follow [WebShop setup](https://github.com/princeton-nlp/WebShop#-setup) to download products, install dependencies, and build the index. Keep those dependencies accessible to the training Python process.

```bash
git clone https://github.com/princeton-nlp/WebShop.git /path/to/webshop
cd /path/to/webshop
bash setup.sh -d small
```

Set `WEBSHOP_ROOT` to that directory. Set `JAVA_HOME` when the JVM is outside the Python environment. The recovered backend defaults to 1,000 products and `human_goals=0`; `WEBSHOP_NUM_PRODUCTS` and `WEBSHOP_HUMAN_GOALS` configure these values. Use the same corpus settings for data preparation, training, and evaluation.

The preparation script retains the recovered goal ordering: the test prefix is `[0, test_size)`, the next 1,000 goals are validation goals, and the remaining goals form the training pool. It samples 2,400 tasks from that training pool using the configured seed. The default test prefix contains 100 tasks. Increasing `--test_size` changes the pool boundary, so regenerate the registry before changing that setting.
