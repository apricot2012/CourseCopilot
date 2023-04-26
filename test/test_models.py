import sys
from src.models import tf_idfExtractor, dpr_Extractor, hyde_Extractor

tf = tf_idfExtractor()
print(tf.infer_relevant_docs(['/home/jackgong/cs-440/study-copilot/data/papertest.txt'], "What are some geographic regions that you can purchase EC2 instances in?"))

dpr = dpr_Extractor()
print(dpr.infer_relevant_docs(['/home/jackgong/cs-440/study-copilot/data/papertest.txt'], "What are some geographic regions that you can purchase EC2 instances in?"))

hyde = hyde_Extractor()
print(hyde.infer_relevant_docs(['/home/jackgong/cs-440/study-copilot/data/papertest.txt'], "What are some geographic regions that you can purchase EC2 instances in?"))