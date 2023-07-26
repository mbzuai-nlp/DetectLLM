
# DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text




## Abstract
With the rapid progress of large language models (LLMs) and the huge amount of text they generated, it becomes more and more impractical to manually distinguish whether a text is machine-generated. Given the growing use of LLMs in social media and education, it prompts us to develop methods to detect machine-generated text, preventing malicious usage such as plagiarism, misinformation, and propaganda. Previous work has studied several zero-shot methods, which require no training data. These methods achieve good performance, but there is still a lot of room for improvement. In this paper, we introduce two novel zero-shot methods for detecting machine-generated text by leveraging the log rank information. One is called DetectLLM-LRR, which is fast and efficient, and the other is called DetectLLM-NPR, which is more accurate, but slower due to the need for perturbations. Our experiments on three datasets and seven language models show that our proposed methods improve over the state of the art by 3.9 and 1.75 AUROC points absolute. Moreover, DetectLLM-NPR needs fewer perturbations than previous work to achieve the same level of performance, which makes it more practical for real-world use. We also investigate the efficiency — performance trade-off based on users preference on these two measures and we provide intuition for using them in practice effectively.

<p align="center" width="100%"><a href="https://github.com/mbzuai-nlp/DetectLLM" target="github">GitHub</a>, <a href="https://arxiv.org/pdf/2306.05540.pdf" target="github">Paper</a></p>



## Citation
Please cite us if you use our data or models.
```bibtex
@article{su2023detectllm,
  title={DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text},
  author={Su, Jinyan and Zhuo, Terry Yue and Wang, Di and Nakov, Preslav},
  journal={arXiv preprint arXiv:2306.05540},
  year={2023}
}
```
























# Instructions

## Create environment and run experiments
```
conda create --name DetectLLM python=3.8 
conda activate DetectLLM
pip install -r requirements.txt

bash run.sh # run bash file
```
