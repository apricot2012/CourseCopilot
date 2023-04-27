import pickle

class Loader:
    def init(self):
        pass

    def load_elements(self, filename, path = "./data"):
        tokenized_path = path + "/tokenized"
        elements_reconstructed = None
        with open(tokenized_path + "/" + filename + ".pickle", "rb") as infile:
            elements_reconstructed = pickle.load(infile)
        return elements_reconstructed