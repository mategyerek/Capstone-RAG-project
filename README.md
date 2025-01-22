# Capstone group 6 code documentation
This file serves as a documentation for the project Leveraging Large Language Models for Policy Document Retrieval information in Energy Renovations. The project is part of the Capstone course for the engineering with AI minor at the TU Delft.
## Steps to reproduce results
This section will describe how to reproduce the results in our presentation. To run the more demanding parts in a reasonable time, GPU inference is recommended which involves special setup and hardware. 
### Setting up the environment
Clone this repository (please make sure to not commit anything, as you can only have read-write access).
Make sure to use python 3.12. Run `pip install -r requirements_loose.txt` (requirements.txt is included for exact versions used, but is probably not portable) to install all the required modules.
Download the data from https://huggingface.co/datasets/vidore/syntheticDocQA_energy_test_ocr_chunk/tree/main. Place the contents (`test-00000-of-00002.parquet` and `test-00000-of-00002.parquet`) into `Capstone-RAG-project/data`.

This should take care of the basic setup, but if any modules are missing just install them with pip.
### Special setup for GPU inference (optional)
To run the LLM inference locally, you need to have a GPU available. Beware that the setup was only tested on Ubuntu 22.04.5 LTS and an nvidia GPU with compute capability 7.5. If you have different os or hardware you might have to deviate from these steps.
You need to have the cuda drivers and cuda toolkit installed and working (setup tested with version 12.1). If you have multiple versions make sure to have the correct one selected by running `nvcc --version`. To select another version run
<code>
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
</code>
Then to install the required packages with CUDA, run:
<code>
export GGML_CUDA=1
CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
pip install llama-cpp-haystack --upgrade --force-reinstall --no-cache-dir
</code>
Note that if you try this on Windows you will probably get an [error](https://github.com/abetlen/llama-cpp-python/issues/721#issuecomment-1723892241).

### Downloading models
If you would like to run inference, the model weights need to be downloaded from huggingface (the models are several GB each, you can choose to omit any of them just make sure to exclude them from the parameter search). In our study the following models were used:
<code>https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf
https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q3_K_L.gguf
https://huggingface.co/S10244414/Mistral-7B-v0.3-Q5_K_M-GGUF/resolve/main/mistral-7b-v0.3-q5_k_m.gguf</code>
The downloaded files should be placed in `Capstone-RAG-project/model_weights`

### Running the code
After completing the steps above, you can run the code. Run the `querydata.py` to extract the relevant part of the data from the raw data. Then run `embed_document.py` to embed the documents.
You can run `NO-RAG Chat.py` to get the baseline performance from Chatgpt-4o. The API key included in source should work, but if it does not, generate your own OpenAI API key.
You can run `eval_auto_pipeline_locallm.py` to evaluate a combination of LLM and embedding model or do a parameter search across multiple ones. Note that this will take a long time especially if its running on the CPU. For the parameter search you have to specify the list of embedding models and LLMs in source. The embedding models are loaded automatically but the LLM's .gguf file should be manually downloaded as described above.
The parameters for the run need to be changed in source. Adjust the variables `embedding_models`, `generator_models`, `temperature` and `repeat_penalty` to reproduce specific runs. Set `test` to true if you want data on the test set. All the results are placed in the results folder.

Finally, run the Data Analysis Notebook to produce the plots from the data.
## Description of program structure and design choices
### File structure
### Choice of framework
### Main difficulties and solutions
### Limitations of the design
