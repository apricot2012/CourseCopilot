from src.ingestor import Ingestor
from src.models import tf_idfExtractor, dpr_Extractor, hyde_Extractor
from os import listdir
from os.path import isfile, join, isdir
from pathlib import Path
import src.evaluator as eval
from src.enquirer import Enquirer

# ingestor = Ingestor()
# ingestor.ingest()

segment_length = 512
top_k = 5


question_path = './data/questions'
extracted_path = './data/extracted'
groundtruth_path = './data/groundtruth'
text_path = './data/tokenized'

text_directories = [t for t in listdir(text_path) if isdir(join(text_path, t))]
for t in text_directories:
    text_file_paths = [join(text_path, t, f) for f in listdir(join(text_path, t))]
    question_files = [join(question_path, t, f) for f in listdir(join(question_path, t)) if isfile(join(question_path, t, f))]
    question_names = [Path(q).stem for q in question_files]

    enquirer = Enquirer()
    tf = tf_idfExtractor(text_file_paths, segment_length, top_k)
    dpr = dpr_Extractor(text_file_paths, segment_length, top_k)
    hyde = hyde_Extractor(text_file_paths, segment_length, top_k)
    #models = [tf, dpr, hyde]
    models = [tf, dpr]

    for i in range(len(question_files)):
        with open(question_files[i]) as f:
            query = f.readline()
        with open(join(groundtruth_path, t, question_names[i] + '.txt')) as f:
            lines = f.readlines()
            groundtruth = eval.lines_to_relevant_strings(lines, tf.docs, segment_length)

        for model in models:
            docs = model.infer_relevant_docs(query)
            docs_as_string = [doc.page_content for doc in docs]
            print(docs_as_string)
            print(eval.percentage_contained(groundtruth, docs_as_string))
            print(enquirer.perform_qa(docs, query))