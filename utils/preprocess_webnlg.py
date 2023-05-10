import json

fixed_intruction="The input is a list of triplets. Each triple has a subject, a predicate and an object. Try to produce a sentence out of it."
def preprocess_webnlg(path):
    with open(path,"r") as file:
        data=file.read()
    obj=json.loads(data)
    entries=obj["entries"]
    output_list=list()
    for i in range(len(entries)):
        for lex in list(map(lambda el: el["lex"],entries[i][str((i+1))]["lexicalisations"])):
            output_list.append({
                "instruction": fixed_intruction,
                "input":entries[i][str((i+1))]["modifiedtripleset"],
                "output": lex
            })
    return output_list

def preprocess_webnlg_val(path):
    with open(path,"r") as file:
        data=file.read()
    obj=json.loads(data)
    entries=obj["entries"]
    output_list=list()
    for i in range(len(entries)):
        output_list.append({
            "instruction": fixed_intruction,
            "input":entries[i][str((i+1))]["modifiedtripleset"],
            "output": list(map(lambda el: el["lex"],entries[i][str((i+1))]["lexicalisations"]))
        })
    return output_list


