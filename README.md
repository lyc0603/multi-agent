# multi-agent
Project to study the multi-agent crypto fund management.

Navigate to the directory of the cloned repo

```bash
cd multi-agent
```

## Installation

To install the latest release on `PyPI <https://pypi.org/project/toml/>`_, run:

```bash
pip install toml
```

### Create a python virtual environment

- iOS

```zsh
python3 -m venv venv
```

- Windows

```
python -m venv venv
```

### Activate the virtual environment

- iOS

```zsh
. venv/bin/activate
```

- Windows (in Command Prompt, NOT Powershell)

```zsh
venv\Scripts\activate.bat
```

## Install the project in editable mode

```
pip install -e ".[dev]"
```
---

# Run

## CoinGecko

- Merge Coingecko chart data
```bash
python scripts/process/signal/gecko_all.py
```

- Process crypto daily data
```bash
python scripts/process/signal/crypto_daily.py
```

- Process common factors
```bash
python scripts/process/signal/common_factors.py
```

- Process weekly data
```bash
python scripts/process/signal/crypto_weekly.py
```

- Process CAPM data
```bash
python scripts/process/signal/capm.py
```

- Process crypto market data
```bash
python scripts/process/signal/cmkt.py
```

## CryptoCompare

- Fetch Cryptocompare data
```bash
python scripts/environ/fetch/cryptocompare.py
```

- Process OHLC data
```bash
python scripts/process/signal/ohlc.py
```


## Environ
- Process the environment data
```bash 
python scripts/process/env_data.py
```