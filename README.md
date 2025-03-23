# Haystack_Huggingface

## 1. Objectives

### 1.1 Try Haystack with HuggingFace Free Serverless Inference

This repo is focused on trying Haystack tutorials, replacing OpenAI inference with HuggingFace

### 1.2 Understand how Haystack functions and objects work

The tutorials explains well how the Haystack's objects and functions purposes and how to works

## 2. Models used

Links to supported models: [Supported HuggingFace Model]('https://huggingface.co/docs/api-inference/en/supported-models)

## 3. Method

### 3.1 Created custom HuggingFace Chat Generator ``custom_HF/chat_generator.py``

### 3.2 Created custom HuggingFace Document Embedder ``custom_HF/doc_embedder.py``

### 3.3 Created custom HuggingFace Text Embedder ``custom_HF/text_embedder.py``

## 4. Outcome

### 4.1 Beginner

|Number|                    Tutorial                   |   Result   | Failure Reason |
| :--: | --------------------------------------------- | :--------: |  :----------:  |
|  01  | 01_QA_Pipeline_Retrieval_Augmentation         | Successful |                |
|  02  | 02_Retrieving_Context_Window_Around_Sentence  | Successful |                |
|  03  | 03_Filtering_Documents_Metadata               | Successful |                |
|  04  | 04_Preprocessing_Different_File_Types         | Successful |                |
|  05  | 05_Embedding_Metadata_Improved_Retrieval      | Successful |                |
|  06  | 06_Serializing_LLM_Pipelines                  | Successful |                |
|  06  | 07_Build_Extractive_QA_Pipeline               | Successful |                |

### 4.2 Intermediate

|Number|                                       Tutorial                                 |   Result   |  Failure Reason  |
| :--: | ------------------------------------------------------------------------------ | :--------: |  :------------:  |
|  01  | 01_Structured_Output_Loop_Based_Auto_Correction                                | Failed     |JSON format invalid [Link](https://github.com/deepset-ai/haystack/discussions/6900)                |
|  02  | 02_Fallbacks_to_Websearch_with_Conditional_Routing                             | Successful |                  |
|  03  | 03_Creating_Hybrid_Retrieval_Pipeline                                          | Failed     |Endpoint Timed Out|
|  04  | 04_Query_Classification_TransformersTextRouter_TransformersZeroShotTextRouter  | Successful |                  |
|  05  | 05_Evaluating_RAG_Pipelines                                                    | Failed |FaithfulnessEvaluator only works with OPENAI [link](https://docs.haystack.deepset.ai/docs/faithfulnessevaluator)                  |
|  06  | 06_Classifying_Documents_Queries_Language                                      | Successful |                  |

### 4.3 Advanced

|Number|                    Tutorial                   |   Result   | Failure Reason |
| :--: | --------------------------------------------- | :--------: |  :----------:  |
|  01  | 01_Chat_Agent_with_Function_Calling           |    Failed  |                |