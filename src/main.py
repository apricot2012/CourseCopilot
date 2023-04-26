from src.ingestor import Ingestor
from src.models import tf_idfExtractor, dpr_Extractor, hyde_Extractor
from os import listdir
from os.path import isfile, join
from pathlib import Path
import src.evaluator as eval
from src.enquirer import Enquirer

ingestor = Ingestor()
ingestor.ingest()

segment_length = 512
top_k = 5

tf = tf_idfExtractor(segment_length, top_k)
dpr = dpr_Extractor(segment_length, top_k)
hyde = hyde_Extractor(segment_length, top_k)
models = [tf, dpr, hyde]

question_path = './data/questions'
extracted_path = './data/extracted'
groundtruth_path = './data/groundtruth'
text_path = './data/state.txt'

question_files = [join(question_path, f) for f in listdir(question_path) if isfile(join(question_path, f))]
question_names = [Path(q).stem for q in question_files]

enquirer = Enquirer()

for i in range(len(question_files)):
    with open(question_files[i]) as f:
        query = f.readline()
    with open(join(groundtruth_path, question_names[i] + '.txt')) as f:
        lines = f.readlines()
        groundtruth = eval.lines_to_relevant_strings(lines, [text_path], segment_length)

    for model in models:
        docs = model.infer_relevant_docs([text_path], query)
        docs_as_string = [doc.page_content for doc in docs]
        print(docs_as_string)
        print(eval.percentage_contained(groundtruth, docs_as_string))
        print(enquirer.perform_qa(docs, query))