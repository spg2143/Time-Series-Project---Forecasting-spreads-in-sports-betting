import modules.models.ensemble as ensemble
test = ensemble.ensemble_wrapper() 
a = 5
b = 2
test.a, test.b = a, b
print(test.add_ab())