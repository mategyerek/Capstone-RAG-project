# Capstone group 6 code documentation
This file serves as a documentation for the project Leveraging Large Language Models for Policy Document Retrieval information in Energy Renovations. The project is part of the Capstone course for the engineering with AI minor at the TU Delft.
## Steps to reproduce results
This section will describe how to reproduce the results in our presentation. To run the more demanding parts in a reasonable time, GPU inference is recommended which requires special setup and hardware.

### Setting up the environment
Clone this repository (please make sure to not commit anything, as you can only have read-write access).

Make sure to use python 3.12. Run `pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary=llama-cpp-python` to install all the required modules. If there is no pre-built llama-cpp-python wheel compatible with your hardware this will fail. In this case resort to the [llama-cpp-python documentation](https://github.com/abetlen/llama-cpp-python/tree/main?tab=readme-ov-file#installation-configuration) for instructions.

Download the data from https://huggingface.co/datasets/vidore/syntheticDocQA_energy_test_ocr_chunk/tree/main. Place the contents (`test-00000-of-00002.parquet` and `test-00000-of-00002.parquet`) into `Capstone-RAG-project/data`.

This should take care of the basic setup, but if any modules are missing just install them with pip.
### Special setup for GPU inference (optional)
To run the LLM inference locally, you need to have a GPU available. Beware that the following setup was only tested on Ubuntu 22.04.5 LTS and an nvidia GPU with compute capability 7.5. If you have different os or hardware you might have to deviate from these steps.

You need to have the cuda drivers and cuda toolkit installed and working (setup tested with version 12.1). If you have multiple versions make sure to have the correct one selected by running `nvcc --version`. To select another version run

<code>export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH</code>

Then to install the required packages with CUDA, run:

<code>export GGML_CUDA=1
CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
pip install llama-cpp-haystack --upgrade --force-reinstall --no-cache-dir</code>

Note that if you try this on Windows you will probably get an [error](https://github.com/abetlen/llama-cpp-python/issues/721#issuecomment-1723892241). In this case I recommend trying a to use a pre-built wheel for your cuda version (something like `pip install -r requirements.txt --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version> --only-binary=llama-cpp-python`), but I haven't tried this so no guarantees.

### Downloading models
If you would like to run inference, the model weights need to be downloaded from huggingface (the models are several GB each, you can choose to omit any of them just make sure to exclude them from the parameter search). In our study the following models were used:

* https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
* https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf
* https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q3_K_L.gguf
* https://huggingface.co/S10244414/Mistral-7B-v0.3-Q5_K_M-GGUF/resolve/main/mistral-7b-v0.3-q5_k_m.gguf

The downloaded files should be placed in `Capstone-RAG-project/model_weights`

### Running the code
After completing the steps above, you can run the code. Run the `querydata.py` to extract the relevant part of the data from the raw data. Then run `embed_document.py` to embed the documents.
You can run `NO-RAG Chat.py` to get the baseline performance from Chatgpt-4o. The API key included in source should work, but if it does not, generate your own OpenAI API key.
You can run `eval_auto_pipeline_locallm.py` to evaluate a combination of LLM and embedding model or do a parameter search across multiple ones. Note that this will take a long time especially if its running on the CPU. For the parameter search you have to specify the list of embedding models and LLMs in source. The embedding models are loaded automatically but the LLM's .gguf file should be manually downloaded as described above.
The parameters for the run need to be changed in source. Adjust the variables `embedding_models`, `generator_models`, `temperature` and `repeat_penalty` to reproduce specific runs. Set `test` to true if you want data on the test set. All the results are placed in the results folder.

Finally, run the Data Analysis Notebook to produce the plots from the data.
## Description of program structure and design choices
The following part gives an overview of our approach and process. It is not needed to reproduce the results, only to get a deeper understanding of the code.
### File structure
The project is structured as follows. In the root folder we have all the python files:
* `after_processing.py` - A temporary solution to remove hallucinations from the LLM answers after they have been generated. Not used anymore.

* `chat_pipeline.py` - Used to generate RAG answers using the GPT4o mini. Handy to create example answers quickly, but does not include an evaluation.

* `custom_component.py` - Our custom haystack component to remove hallucinations in the main pipeline.

* `Data Analysis Notebook.ipynb` - Jupyter notebook to visualize results. Does not generate the results by itself, only reads csv-s in results folders.

* `embed_document.py` - Embeds the document database. Includes utility functions to load and save the database in `data/`.

* `eval_auto_pipeline_locallm.py` - Runs and evaluates the main pipeline using local LLM inference. Can be used for parameter optimization. Saves results into the results folder.

* `eval_auto_pipeline.py` - Somewhat deprecated but still functional version of `eval_auto_pipeline_locallm.py`. Instead of local LLM inference, it uses Huggingface API calls.

* `inspect_object.py` - Inspect a pickled object for debugging.

* `No_RAG_Chat.py` - Used to generate and evaluate answers by calling ChatGPT without RAG. Its results used as a baseline.

* `querydata.py` - Initial preprocessing of the data. Takes the relevant fields (query, answer, text_content) from the provided `.parquet˛` files, merges tex chunks related to the same document together and saves the data in `.json` format into `data/` for further processing.


The folders are used as follows:
* `data/` stores the initial raw data and the data used for intermediate steps, namely
	- `querys.json`, an array of all questions in order
	- `answers.json`, an array of all answers in order
	- `doc_lookup.json`, a dictionary linking the unique document index to the original
	- `DocMerged_<embedding model name>.json`, file storing the document embeddings made by a certain model
* `model_weights/` stores the .gguf files for local LLMs
* `results/` stores the .csv files for the results in the following format
	- `results_<embedding model>_<LLM>_<temperature>_<repeat penalty>`
* `results_<experiment name>` folders are used to save out the results for a certain experiment where they would get overwritten
* `test_data` is a json of all entries in the test set (only used to make sure that our split did not change)

There are some additional files for the environment configuration as well as this README.

### Choice of framework
Ads our main NLP framework we choose [Haystack](https://docs.haystack.deepset.ai/docs/intro) by Deepset AI. This is a high-level framework based on the [transformers](https://pypi.org/project/transformers/) library. Haystack was the ideal choice for us, because it has a very good abstractions for quickly creating and iterationg on modular pipelines, which was essential to reach outr requirements.

For local inference we decided to use llamacpp, because we were vram constrained. Llamacpp it allows loading only part of the model into vram, while still efficiently utilizing the GPU. Moreover it is very easy to use with the quantized .gguf files available on huggingfacce for a large selection of models.

The rest of the packages we used are pretty standard, like pandas and numpy for basic data manipulation. These were chosen because they are easy to use and well-supported.

### Main difficulties and solutions
During the coding we encountered a number of problems. When we wanted to merge text chunks by document, we ran into the error of duplicated documents (because now multiple questions ha the exact same document belonging to them). When embedding these documents we simply skipped the duplicates, however by doing this we lost the information about which question is paired to which document. This was later resolved by creating a dictionary in `querydata.py` storing these relationships.

We decided to limit our runs to local inference to force ourselves to use less resources, thus making our solution more sustainable. However, it was not easy to install llama-cpp-python with CUDA support as it would not use the GPU or on even break during installation. To fix the latter, we changed to using Linux, fixing the former led us to discover an error in [haystack documentation](https://docs.haystack.deepset.ai/docs/llamacppgenerator) about setting the right environment variables that is now patched thanks to our contribution.

When we got the pipeline working we discovered that the LMM was hallucinating a lot, almost always follow up questions and answers. As these always adhered to the same format we just instructed the LLM to stop generating after the string "Question:". This did not only improve our results a lot, but also sped up the runtime slightly.

### Limitations of the design

The main performance limitations of the currant pipeline unfortunately is the data itself. The text-content chunks were extracted by OCR (Optical Character Recognition) and sometimes crucial information is missing or the extracted data is unintelligible. This could be fixed by using the PDF-s directly, like more advanced [ColPali](https://arxiv.org/abs/2407.01449). However as seen in the paper, running (and training) a vision LLM uses much more resurces than our solution.

Another limitation stems from using the Haystack framework. Haystack's Document object, thats used to store our documents and embeddings is non-serializable, therefore we can not parallelize our pipeline to process multiple queries at once. This might soon be fixed by the introduction of [async pipelines](https://github.com/deepset-ai/haystack/issues/6012).

Lastly, we are of course limited by LLM performance. The small quantized models we managed to run do not reach the quality of full-size SOTA LLMs like GPT4 or R1. However, in return our pipeline can run on a personal computer, as opposed to million-dollar servers.
