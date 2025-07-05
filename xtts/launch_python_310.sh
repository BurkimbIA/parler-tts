sudo add-apt-repository --remove ppa:ubuntugis/ppa
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev python3.10-distutils
python3.10 -m venv mon_env_310
source mon_env_310/bin/activate
