from os import listdir
from os.path import isfile, join, splitext
import pickle
from unstructured.partition.pdf import partition_pdf
from tqdm import tqdm

class Ingestor:
    file_list_name = "files.list"

    def init(self):
        pass

    def ingest(self, path = "../data"):
        raw_path = path + "/raw"
        tokenized_path = path + "/tokenized"

        raw_files = [f for f in listdir(raw_path) if isfile(join(raw_path, f))]
        split_by_type = dict()
        print("collecting filenames...")
        for file in tqdm(raw_files):
            ext = splitext(file)[-1].lower()
            if ext not in split_by_type:
                split_by_type[ext] = list()
            split_by_type[ext].append(file)

        with open(tokenized_path + "/" + self.file_list_name, "wb") as outfile:
            pickle.dump(split_by_type, outfile)

        # Only handle .pdf files for now
        pdfs = sorted(split_by_type[".pdf"])
        print("ingesting pdfs...")
        for file in tqdm(pdfs):
            self.ingest_pdf(file)
        


    def ingest_pdf(self, filename, path = "../data"):
        raw_path = path + "/raw"
        tokenized_path = path + "/tokenized"

        elements = partition_pdf(raw_path + "/" + filename)
        with open(tokenized_path + "/" + filename + ".pickle", "wb") as outfile:
            pickle.dump(elements, outfile)
