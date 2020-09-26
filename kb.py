# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/9/26
@function: get entity in train data as kb
"""
import json
from collections import defaultdict

type_dict=json.load(open("entity_type.json",encoding="utf8"))
entities=defaultdict(set)

with open("./task1_public/new_train.json",encoding="utf8") as f:
    for line in f:
        item=json.loads(line.strip())
        ent_=item["entities"]
        for ent in ent_:
            if ent["type"]=="Drug":
                entities["Drug"].add(ent["entity"])
            elif ent["type"]=="Disease":
                entities["Disease"].add(ent["entity"])
            elif ent["type"]=="Gene":
                entities["Gene"].add(ent["entity"])
            elif ent["type"]=="ChemicalCompound":
                entities["ChemicalCompound"].add(ent["entity"])
            elif ent["type"]=="Virus":
                entities["Virus"].add(ent["entity"])
            elif ent["type"]=="Chemical":
                entities["Chemical"].add(ent["entity"])
            elif ent["type"]=="Phenotype":
                entities["Phenotype"].add(ent["entity"])
            else:
                entities["Organization"].add(ent["entity"])

entities={key:list(val_set) for key,val_set in entities.items()}

json.dump(entities,open("entity_kb.json","w",encoding="utf8"),ensure_ascii=False,indent=4)
