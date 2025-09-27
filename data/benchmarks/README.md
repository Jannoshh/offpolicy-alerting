# Math Benchmarks

`scripts/download_math_benchmark.py` loads the GSM8K dataset directly from the Hugging Face hub and can generate answers with a language model (requires `torch`).

```bash
python scripts/download_math_benchmark.py --subset-size 50 --show 2
```

The command above reports split sizes, previews a couple of reference solutions, and then samples answers for 50 `test` examples using `sshleifer/tiny-gpt2`. Override `--model`, `--target-split`, or sampling parameters to experiment with different setups.
