import sys
from src.models import tf_idfExtractor, dpr_Extractor, hyde_Extractor

tf = tf_idfExtractor()
print(tf.infer_relevant_docs(['/home/jackgong/cs-440/study-copilot/data/state.txt'], "Who is the president?"))