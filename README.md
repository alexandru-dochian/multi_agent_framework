## Multi Agent Framework

## Installation:

### Recommended tools:

- [Visual Studio Code](https://code.visualstudio.com/download)
- [PyCharm](https://www.jetbrains.com/pycharm/download/)

### Setup

Install all the necessary operating system dependencies and set up python virtual environment with `poetry`.

- [Python 3.11](https://www.python.org/downloads/release/python-3110/)
- [Docker](https://docs.docker.com/get-docker/)
- [Poetry](https://python-poetry.org/docs/#installation)

### Ubuntu

```bash
./setup.sh
```

#### Windows 10

In PowerShell

```shell
```

#### MacOS

TODO

### Run experiment

#### Start Redis communicator

```shell
docker compose up

```

#### Start experiment

Config files in json format are used to start an experiment. They are stored under `maf/config/` directory.

```bash
# No args => it will automatically use `config/maf/default.json`
python3 main.py 
```

New config `new_config.json` should be added in `maf/config/` directory.

Thus `maf/config/new_config.json` can be used as follows:

```bash
python3 main.py maf/config/new_config.json
```

### On-the-edge deployment of crazyflie drones

![DevelopmentLayer](docs/static-resources/DevelopmentLayer.png)