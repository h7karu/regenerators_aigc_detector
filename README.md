# regenerators_aigc_detector
This repository contains team regenerator's Artificial Intelligence Generated Content (AIGC) detector for TikTok TechJam 2026.

## Project overview

## Setup and installation instructions

### how to download cifake dataset:
btw im using wsl ubuntu, not sure if same for mac or windows powershell etc

1. update and set up your local git repo
```
git pull origin main
```
add in the .env file in the repo root directory.
the .env file contains super secret API keys and SHOULD NOT BE PUSHED

2. set up virtual environment
in the terminal:
```
sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate
```
you should see your terminal change from
```
zecha@zlaptop:~/Hackathons/TikTokTechJam26/regenerators_aigc_detector$
``` 
to:
```
(venv) zecha@zlaptop:~/Hackathons/TikTokTechJam26/regenerators_aigc_detector$ 
```
which means ur inside the virtual environment (wtv that means)

3. install required packages
in your terminal:
```
pip install -r requirements.txt
```
pip install will look at requirements.txt and install all required packages.
requirements.txt may be updated in the future when new packages are needed. if so, u can just run pip install again

4. 'compile' and run script
from repo root directory:
```
chmod +x scripts/download_cifake.sh
./scripts/download_cifake.sh
```
If you see `cannot execute: required file not found` on WSL, the script may have Windows (CRLF) line endings. 
Run the command below before trying again.

```
sed -i 's/\r$//' scripts/download_cifake.sh
```

data should be installed under data/cifake with its correct subfolders 
- train
    - REAL
    - FAKE
- test
    - REAL
    - FAKE

## Steps to reproduce the results

## Limitations
A brief reflection on your solution's limitations and what you would improve given more time
## Contribution
Team member contributions (if applicable, i.e. team participants, non-solo participants)
