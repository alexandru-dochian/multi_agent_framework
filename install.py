import os
import platform
import subprocess


def install_ubuntu():
    def run_cmd(cmd: str):
        os.system(cmd)

    print("Installing on Ubuntu..")
    run_cmd("sudo add-apt-repository ppa:deadsnakes/ppa -y")
    run_cmd(
        'sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"'
    )
    run_cmd("sudo apt-get update")

    print("Install python3.11")
    run_cmd("sudo apt install python3.11")

    print("Install Poetry")
    run_cmd("curl -sSL https://install.python-poetry.org | python3 -")
    run_cmd("pip install poetry-plugin-export")

    print("Install Docker")
    run_cmd(
        "sudo apt-get install apt-transport-https ca-certificates curl software-properties-common -y"
    )
    run_cmd(
        "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -"
    )
    run_cmd("sudo apt-get install docker-ce -y")
    run_cmd(f"sudo usermod -aG docker {os.getlogin()}")

    print("Configure poetry environment")
    run_cmd("poetry env use 3.11")
    run_cmd("poetry install")
    run_cmd("poetry shell")
    run_cmd("python3 -m pip install --upgrade pip")

    print("Install crazyflie-lib-python for crazyflie drones")
    run_cmd("git clone https://github.com/bitcraze/crazyflie-lib-python")
    run_cmd("cd crazyflie-lib-python && pip install -e . && cd ..")
    run_cmd("python3 -m pip install cfclient")

    print("Install PyQt5")
    run_cmd("poetry run pip install pyqt5")

    run_cmd("deactivate > /dev/null")

    print("Setting up USB permissions for Crazyradio dongle")
    run_cmd("sudo groupadd plugdev")
    run_cmd(f"sudo usermod -a -G plugdev {os.getlogin()}")

    # Create and write the udev rules to /etc/udev/rules.d/99-bitcraze.rules
    udev_rules = """
    # https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/usb_permissions/
    # Crazyradio (normal operation)
    SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="7777", MODE="0664", GROUP="plugdev"
    # Bootloader
    SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="0101", MODE="0664", GROUP="plugdev"
    # Crazyflie (over USB)
    SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0664", GROUP="plugdev"
    # Debugger (https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/development/openocd_gdb_debugging/)
    SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="3748", MODE="0664", GROUP="plugdev"
    """

    with open("/tmp/99-bitcraze.rules", "w") as file:
        file.write(udev_rules)

    run_cmd("sudo cp /tmp/99-bitcraze.rules /etc/udev/rules.d/99-bitcraze.rules")

    print("Reload udev rules to apply changes")
    run_cmd("sudo udevadm control --reload-rules")
    run_cmd("sudo udevadm trigger")


def install_windows(auto_detect=False):
    def install_windows_with_powershell():
        def run_cmd(cmd: str):
            subprocess.run(["powershell", "-Command", cmd], shell=True)

        print("Installing on Windows using PowerShell..")

        run_cmd(
            "(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -"
        )
        # TODO
        ...

    def install_windows_with_wsl():
        def run_cmd(cmd: str):
            subprocess.run(["wsl", "bash", "-c", cmd], shell=True)

        print("Installing on Windows using WSL..")

        run_cmd("curl -sSL https://install.python-poetry.org | python3 -")
        # TODO
        ...

    if auto_detect:

        def is_wsl():
            # # TODO: To be validated
            # Check for the presence of the WSL environment variable
            return "WSLENV" in os.environ

        def is_powershell():
            return os.environ.get("PSModulePath") is not None

        if is_wsl():
            install_windows_with_wsl()
        elif is_powershell():
            install_windows_with_powershell()
        else:
            raise Exception(
                "Could not detect whether script is being run from wsl or powershell"
            )

    else:
        choice = (
            input("Specify installation agent: 'wsl' or 'powershell': ").strip().lower()
        )
        if choice == "wsl":
            install_windows_with_wsl()
        elif choice == "powershell":
            install_windows_with_powershell()
        else:
            raise ValueError("Invalid input. Please enter 'wsl' or 'powershell'.")


def install_macos():
    def run_cmd(cmd: str):
        os.system(cmd)

    print("Installing on MacOS.")

    run_cmd("curl -sSL https://install.python-poetry.org | python3 -")
    # TODO


def main():
    current_platform = platform.system()

    if current_platform == "Linux":
        distro = subprocess.run(
            ["lsb_release", "-is"], capture_output=True, text=True
        ).stdout.strip()
        if distro in ["Ubuntu", "Debian"]:
            install_ubuntu()
        else:
            print(f"Unsupported Linux distribution: {distro}")
    elif current_platform == "Windows":
        install_windows()
    elif current_platform == "Darwin":
        install_macos()
    else:
        print(f"Unsupported platform: {current_platform}")


if __name__ == "__main__":
    main()
