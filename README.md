This repository contains the code for generating and evaluating instructions using Alpaca and GPT-4 for Business Process Optimization (CLMs4BPO). The main script, instruction_improvement_loop.py, orchestrates the process of generating instructions, evaluating them, and tracking the results.


## Clone repo

```bash
git clone https://github.com/lorenzobagnol/alpaca-lora-CLMs4BPO
```

### Run locally
For this code I used python 3.10. To run the script you need to (create a virtual environment and) install the python libraries from the requirements.txt
```bash
pip install -r requirements.txt
```

### Run in Docker container
Build the image from the Dockerfile
```bash
docker build -t alpaca-image .
```
Then run the container 
```bash
docker run --name alpaca-container --gpus all -v .:/home/workspace/alpaca-CLMs4BPO -it alpaca-image
```

Now you are inside the Docker container ready to launch the app.