# Multi Agent Framework

## Supported operating systems

<p align="center">
  <a href="https://ubuntu.com/desktop">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Ubuntu-logo-no-wordmark-solid-o-2022.svg/640px-Ubuntu-logo-no-wordmark-solid-o-2022.svg.png" alt="Ubuntu" width="auto" height="50" />  &nbsp;&nbsp;&nbsp;&nbsp;
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://support.apple.com/macos">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/MacOS_logo_%282017%29.svg/640px-MacOS_logo_%282017%29.svg.png" alt="MacOS" width="auto" height="50" />
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.microsoft.com/software-download">
    <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Windows_10_Logo.svg" alt="Windows 10" width="auto" height="50" />
  </a>
</p>

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

## Recommended tools:

<p align="center">
  <a href="https://www.jetbrains.com/pycharm/download/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/JetBrains_PyCharm_Product_Icon.svg/640px-JetBrains_PyCharm_Product_Icon.svg.png" alt="PyCharm" width="auto" height="50" />
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://code.visualstudio.com/download">
    <img src="https://code.visualstudio.com/assets/favicon.ico" alt="Visual Studio Code" width="50" height="50" />
  </a>
</p>

## Installation

```shell
python install.py
```

## Play

### Start Redis communicator

```shell
docker compose up

```

### Start experiment

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

## Architecture

![Architecture](docs/static-resources/V9-Architecture.drawio.svg)

## Experiments

### Hello World experiment (`hello_world.json`)

![Hello World](docs/static-resources/V10-Architecture-hello_world.drawio.svg)

### Default experiment (`default.json`)

![Default](docs/static-resources/V9-Architecture-default.drawio.svg)

### Circle around center

(Experiment was conducted in VU Amsterdam's laboratory)

[![Watch the video](docs/static-resources/circle_around_center_vu_thumbnail.png)](https://alexandru-dochian.github.io/multi_agent_framework/pages/circle_around_center.html)

### Circle spin

(Experiment was conducted in VU Amsterdam's laboratory)

[![Watch the video](docs/static-resources/circle_spin_vu_thumbnail.png)](https://alexandru-dochian.github.io/multi_agent_framework/pages/circle_spin.html)

## Integrations

### Crazyflie

[Bitcraze documentation](https://www.bitcraze.io/documentation/repository/)

![Crazyflie integration](docs/static-resources/V9-Architecture-crazyflie.drawio.svg)

## Future work

### On-the-edge deployment of crazyflie drones

![DevelopmentLayer](docs/static-resources/V10-OnTheEdge.drawio.svg)

### Cloud centric extension

![DevelopmentLayer](docs/static-resources/V11-Cloud_Centric.drawio.svg)

## LICENSE

This project is released under [AGPLv3](https://www.gnu.org/licenses/agpl-3.0.txt)
and it applies specifically to the [**maf**](maf) package.

## Reference

Alexandru Dochian. (2024). *Multi Agent Framework for Collective Intelligence Research*. arXiv preprint arXiv:2408.12391. [https://arxiv.org/abs/2408.12391](https://arxiv.org/abs/2408.12391)

