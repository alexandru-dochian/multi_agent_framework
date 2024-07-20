## Drone Framework

### Setup

Install all the necessary operating system dependencies and set up python virtual environment with `poetry`.

Currently supporting:

- Ubuntu
- MacOS (TODO)
- Windows 10 (TODO)

```bash
./setup.sh
```

### Run experiment

Config files in json format are used to start an experiment. They are stored under `config/` directory.

```bash
# No args => it will automatically use `config/default.json`
python3 main.py 
```

New config `new_config.json` should be added in `config/` directory.
Thus `config/new_config.json` can be used as follows:

```bash
python3 main.py new_config.json
```

### On-the-edge deployment of crazyflie drones

![DevelopmentLayer](docs/static-resources/DevelopmentLayer.png)