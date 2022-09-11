# Abstract
 
Document summarization is a very useful tool for quickly going through huge amount of information and understanding the essence. However, because people have different interests for each individual, if the summary is generated based on a different interest than you have, you can not find the information that you are looking for. In this study, we propose a system that generates topic-by-topic summary from a document consisting of multiple topics. To deal with this type of problem, we create a new dataset annotated with a summary for each topic. The dataset is created from integrated reports, which contain text labeled with each goal of the SDGs (Sustainable Development Goals), and we consider each goal as a topic. We solve this problem as a query-focused extractive text summarization problem and develop a novel query-focused summarization method for the created dataset. For comparison purpose, we devise a solution to solve this problem as a Question Answering (QA) task and we compare the performance of query-focused summarization methods, QA task methods, and several existing baseline methods. As a result of experiment, our proposed method of query-focused summarization outperforms existing baseline methods by 30\% and outperforms the QA task methods by 11%. 

See the presentation pack of the research from [here](https://www.slideshare.net/ssuserf66333/queryfocused-extractive-text-summarization-for-multitopic-document).

# Dataset
We take advantage of integrated reports as the source of our dataset. An integrated report is a report issued by a company for investors on an annual basis that integrates financial information, with non-financial information, such as environmental and social initiatives. Some of the integrated reports have labels to indicate relevance between their initiatives and the 17 SDGs goals. These integrated reports are not only suitable as multi-topic documents, but also can be seen as a corpus with labels of the 17 SDGs already annotated by corporate IRs.

See the list of links to the original Integrated Reports from [here](https://github.com/shinichiromizuno/QueryMultiTopic/blob/master/List_of_Links_to_Inregrated_Report.txt).
 
# Usage

1. Download the source codes in your Google Drive (git clone).
2. Create empty working directories for each model (such as work_BERT_Base) in your Google Drive.
3. Open Collab Notebooks in your Google Colaboratory and execute step by step.
   Due to the dependency of dataset preprocessing, you may want to execute the notebooks in the following order;
   BERT_Base.ipynb -> Multi_BERTSum.ipynb -> Any other notebooks
 
# Note

The original source code was borrowed from [BertSum](https://github.com/nlpyang/BertSum). 