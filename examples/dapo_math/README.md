# Math

Prepare a deterministic 4,000-example sample from DAPO-Math-17k, plus AIME 2024 and 2025 validation data:

```bash
python -m examples.dapo_math.prepare_math_data --train_size 4000 --seed 42
bash examples/dapo_math/train_dapo_math.sh
```

`--train_dataset`, `--aime24_dataset`, and `--aime25_dataset` accept Hugging Face IDs or local dataset paths.
