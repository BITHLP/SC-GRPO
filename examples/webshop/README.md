# WebShop

SC-GRPO runs the WebShop backend in-process. Set `WEBSHOP_ROOT` to your checkout and prepare the data before training. The default preparation uses 2,400 randomly sampled training tasks and the recovered test prefix of 100 tasks; `--test_size` configures that prefix.

```bash
python -m examples.webshop.prepare_webshop_data --train_size 2400 --seed 42 --dataset_name webshop_text_official
bash examples/webshop/train_webshop_agent.sh
```

The backend product count defaults to 1,000. Set `WEBSHOP_NUM_PRODUCTS` consistently during preparation and training to use a different installed product corpus. See [environment setup](../../docs/environments.md).
