sudo add-apt-repository --remove ppa:ubuntugis/ppa
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev python3.10-distutils
python3.10 -m venv mon_env_310
source mon_env_310/bin/activate
git clone https://github.com/BurkimbIA/toolskit-tts.git
cd toolskit-tts/xtts
pip install -r requirements.txt
python generate_data.py
python extend_vocab_config.py --output_path=/content/drive/MyDrive/checkpoints --metadata_path dataset/metadata_train.csv --language mos --extended_vocab_size 2000

