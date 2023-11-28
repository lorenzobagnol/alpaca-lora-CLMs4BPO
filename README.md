Il file principale è prompt_generator.py. 
Lo script segue i seguenti passaggi:
- carica il dataset
- legge il file initial_instruction_evaluations.csv che contiene le generazioni di Alpaca sul nostro dataset e le loro valutazioni (fatte da GPT-4).
- avvia un loop in cui ad ogni iterazione GPT-4 genera una nuova instruction, Alpaca usa questa nuova instruction per generare del testo sui dati del dataset, GPT-4 valuta queste generazioni. 

Il file GPT-4_loop_results.csv tiene traccia di tutte le instruction generate in questo loop e delle loro valutazioni.
Il file GPT-4_best_instruction_evaluations.csv contiene le generazioni fatte da Alpaca usando la migliore instruction ottenuta dal loop.

I file generate.py e evaluate_gpt4.py contengono rispettivamente la funzione che genera testo dal dataset con Alpaca, e la funzione che valuta le generazioni usando GPT-4.