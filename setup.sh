detect_os() {
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source /etc/os-release
    if [[ "$ID" == "ubuntu" ]]; then
        echo "Ubuntu"
    else
        echo "$PRETTY_NAME"
    fi
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS"
  elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Windows"
  else
    echo "Unknown"
  fi
}

setup_ubuntu() {
  echo "Setting up for Ubuntu..."
  python3 -m pip install --upgrade pip

  echo "Install python3.11"

  sudo add-apt-repository ppa:deadsnakes/ppa -y
  sudo apt update
  sudo apt install python3.11

  echo "Install poetry"
  #  https://python-poetry.org/docs/#installing-with-the-official-installer
  curl -sSL https://install.python-poetry.org | python3 -
  poetry env use 3.11
  poetry install

  echo "Install crazyflie-lib-python for crazyflie drones"
  git clone https://github.com/bitcraze/crazyflie-lib-python
  cd crazyflie-lib-python && pip install -e . && cd ..
  python3 -m pip install cfclient

  poetry shell

  echo "Install pyqt5 for mayavi rendering"
  python3 -m pip install pyqt5
}

setup_macos() {
  echo "TODO: Setting up for macOS..."
}

setup_windows() {
  echo "TODO: Setting up for Windows..."
  # Following line installs poetry. Works only in powershell
  # https://python-poetry.org/docs/#installing-with-the-official-installer
  # (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
}

# Main script logic
OS=$(detect_os)
echo "Detected OS: $OS"

case "$OS" in
  "Ubuntu")
    setup_ubuntu
    ;;
  "macOS")
    setup_macos
    ;;
  "Windows")
    setup_windows
    ;;
  *)
    echo "No support for: $OS"
    exit 1
    ;;
esac
