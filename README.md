# VMH (Video Misleading Headline Dataset)

This is a repository that contains VMH dataset and codes for baseline reproduction from EMNLP 2023 paper that can be found [here](https://arxiv.org/abs/2310.13859)! 

# Abstract
Polarization and the marketplace for impressions have conspired to make navigating information online difficult for users, and while there has been a significant effort to detect false or misleading text, multimodal datasets have received considerably less attention. To complement existing resources, we present multimodal Video Misleading Headline (VMH), a dataset that consists of videos and whether annotators believe the headline is representative of the video's contents. After collecting and annotating this dataset, we analyze multimodal baselines for detecting misleading headlines. Our annotation process also focuses on why annotators view a video as misleading, allowing us to better understand the interplay of annotators' background and the content of the videos.

# Crowdsourcing Framework
The below framework is specifically devised to reduce the subjectivity of the misleading video headline detection task. The annotators are asked to encounter each schemes to finalize their label annotations on the misleadingness of the headline and rationale annotations that demonstrates the reason behind their decisions. 

## Label Annotation Scheme<br>
![label](https://github.com/yysung/VMH/assets/45355533/e34661c3-4a08-49c9-ba29-1026b6560cd9)


<br>

## Rationale Annotation Scheme<br>
![rationale](https://github.com/yysung/VMH/assets/45355533/8b263e1e-51a6-40d8-b5b0-2b7ddd5bffaf)


<br>

# Codebase
The preprocessing code is based on raw data collected from MTurk crowdsourcing platform.

The code for benchmark models ([VideoCLIP](https://aclanthology.org/2021.emnlp-main.544/) and [VLM](https://aclanthology.org/2021.findings-acl.370/)) are forked from https://github.com/facebookresearch/fairseq/blob/main/examples/MMPT/README.md
Both models are video-text retrieval models, which are added a classification layer to be used a classifier that detects misleading video headlines. 
Code is available in ./scripts/MultimodalBaselines.


