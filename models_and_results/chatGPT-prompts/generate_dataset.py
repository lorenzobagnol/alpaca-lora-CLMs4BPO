import json

def preprocess_webnlg(path, set):
    with open("/home/bagnol/progetti/NLG/alpaca-lora/chatGPT-prompts/prompts.txt","r") as file:
        prompts=file.read().splitlines()
    with open(path,"r") as file:
        data=file.read()
    obj=json.loads(data)
    entries=obj["entries"]
    output_list=list()
    lenght=len(entries)
    if set=="train":
        for j in range(3):
            for i in range(len(entries)):
                for lex in list(map(lambda el: el["lex"],entries[i][str((i+1))]["lexicalisations"])):
                    output_list.append({
                        "instruction": prompts[i%len(prompts)-j],
                        "input": entries[i][str((i+1))]["modifiedtripleset"],
                        "output": lex
                    })
    if set=="val":
        for i in range(len(entries)):
            for lex in list(map(lambda el: el["lex"],entries[i][str((i+1))]["lexicalisations"])):
                output_list.append({
                    "instruction": prompts[i%len(prompts)],
                    "input": entries[i][str((i+1))]["modifiedtripleset"],
                    "output": lex
                })
    if set=="test":
        for i in range(len(entries)):
            output_list.append({
                "instruction": prompts[i%len(prompts)],
                "input": entries[i][str((i+1))]["modifiedtripleset"],
                "output": list(map(lambda el: el["lex"],entries[i][str((i+1))]["lexicalisations"]))
            })       
    with open("/home/bagnol/progetti/NLG/alpaca-lora/chatGPT-prompts/dataset/"+set+".json", "w") as file:
        json.dump(output_list, file, indent=4)
        

dataset_base_path="/home/bagnol/progetti/NLG/alpaca-lora/webnlg-dataset-master/"
dataset_version_path="release_v2.1/json/"
preprocess_webnlg(dataset_base_path+dataset_version_path+"webnlg_release_v2.1_train.json","train")
preprocess_webnlg(dataset_base_path+dataset_version_path+"webnlg_release_v2.1_dev.json","val")
preprocess_webnlg(dataset_base_path+dataset_version_path+"webnlg_release_v2.1_test.json","test")