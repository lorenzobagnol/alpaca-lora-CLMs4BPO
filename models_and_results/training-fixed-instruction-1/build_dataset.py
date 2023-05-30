import json

def preprocess_webnlg(path, set):
    with open(path,"r") as file:
        data=file.read()
    obj=json.loads(data)
    entries=obj["entries"]
    output_list=list()
    if set!="test":
        for i in range(len(entries)):
            for lex in list(map(lambda el: el["lex"],entries[i][str((i+1))]["lexicalisations"])):
                output_list.append({
                    "instruction": "The input is a list of triplets. Each triple has a subject, a predicate and an object. Try to produce a sentence out of it.",
                    "input":entries[i][str((i+1))]["modifiedtripleset"],
                    "output": lex
                })
    else:
        for i in range(len(entries)):
            output_list.append({
                "instruction": "The input is a list of triplets. Each triple has a subject, a predicate and an object. Try to produce a sentence out of it.",
                "input":entries[i][str((i+1))]["modifiedtripleset"],
                "output": list(map(lambda el: el["lex"],entries[i][str((i+1))]["lexicalisations"]))
            })       
    with open("/home/bagnol/progetti/NLG/alpaca-lora/training-fixed-instruction-1"+"/dataset/"+set+".json", "w") as file:
        json.dump([ob for ob in output_list], file)
        
        
dataset_base_path="/home/bagnol/progetti/NLG/alpaca-lora/webnlg-dataset-master/"
dataset_version_path="release_v2.1/json/"
preprocess_webnlg(dataset_base_path+dataset_version_path+"webnlg_release_v2.1_train.json","train")
preprocess_webnlg(dataset_base_path+dataset_version_path+"webnlg_release_v2.1_dev.json","val")
preprocess_webnlg(dataset_base_path+dataset_version_path+"webnlg_release_v2.1_test.json","test")
