import joblib
def save_model(model_to_save,path):
    joblib.dump(model_to_save,path)

def load_model(bin_model_path):
    return joblib.load(bin_model_path)