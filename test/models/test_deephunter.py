from deephunter.models import get_net 

import pytest 

def test_get_net():
    name_list = [
        "convmnist",
        # "convcifar10",
        # "vgg",
        # "alexnet",
        # "alexnet",
        ]
    
    for name in name_list:
        model = get_net(name=name)
        assert model is not None 
        

def test_get_official_dataset():
    pass 
def test_get_manual_dataset():
    pass 

def test_create_manual_dataset():
    pass 