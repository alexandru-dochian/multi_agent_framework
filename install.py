import os
import platform
import subprocess


def install_ubuntu():
    def run_cmd(cmd: str):
        os.system(cmd)

    print("Installing on Ubuntu..")
    run_cmd("curl -sSL https://install.python-poetry.org | python3 -")


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
            # # TODO: To be validated
            # Check the shell environment variables
            # PowerShell's `$PSModulePath` variable can be checked (usually in Windows)
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
