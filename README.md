# Multi Agent Framework

## Recommended tools:

<div style="display: flex; justify-content: space-around; align-items: center; padding: 20px;">
  <a href="https://www.jetbrains.com/pycharm/download/" style="text-decoration:none;">
    <img src="https://www.jetbrains.com/favicon.ico" alt="PyCharm" width="50" height="50"/>
  </a>

  <a href="https://code.visualstudio.com/download" style="text-decoration:none;">
    <img src="https://code.visualstudio.com/assets/favicon.ico" alt="Visual Studio Code" width="50" height="50"/>
  </a>
</div>

## Required dependencies

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3110/">
    <img src="https://www.python.org/static/favicon.ico" alt="Python 3.11" width="50" height="50" />
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://docs.docker.com/get-docker/">
    <img src="https://www.docker.com/favicon.ico" alt="Docker" width="50" height="50" />
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://python-poetry.org/docs/#installation">
    <img src="https://python-poetry.org/images/favicon-origami-32.png" alt="Poetry" width="50" height="50" />
  </a>
</p>

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

## LICENCE

This project is licensed under [AGPLv3](https://www.gnu.org/licenses/agpl-3.0.txt)

