import masala_utils

model=masala_utils.recompile_model("weights_utils.hdf5")

print("Ladue Love Calculator.")
print("To exit, leave both name fields blank.\n")

n1=n2="~"

while n1!="" and n2!="":
	n1=raw_input("First Name: ")
	n2=raw_input("Second Name: ")

print(masala_utils.use_model(model, "names.txt", n1, n2))
