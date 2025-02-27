Il file principale è prompt_generator.py. 
Lo script segue i seguenti passaggi:
- carica il dataset
- legge il file initial_instruction_evaluations.csv che contiene le generazioni di Alpaca sul nostro dataset e le loro valutazioni (fatte da GPT-4).
- avvia un loop in cui ad ogni iterazione GPT-4 genera una nuova instruction, Alpaca usa questa nuova instruction per generare del testo sui dati del dataset, GPT-4 valuta queste generazioni. 

Il file GPT-4_loop_results.csv tiene traccia di tutte le instruction generate in questo loop e delle loro valutazioni.
Il file GPT-4_best_instruction_evaluations.csv contiene le generazioni fatte da Alpaca usando la migliore instruction ottenuta dal loop.

I file generate.py e evaluate_gpt4.py contengono rispettivamente la funzione che genera testo dal dataset con Alpaca, e la funzione che valuta le generazioni usando GPT-4.

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