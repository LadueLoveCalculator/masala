import llc_utils

X, y=llc_utils.generate_data('names.txt')

llc_utils.save_model_weights_to_filename((llc_utils.train_model(llc_utils.create_model(), X, y)), "weights_utils.hdf5")