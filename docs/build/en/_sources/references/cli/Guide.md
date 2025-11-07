# CLI Guide
The rtp-llm command-line tool is used to run and manage RTP-LLM. You can start by viewing the help message with:

rtp-llm --help

Available Commands:

rtp-llm {serve}


## serve

Starts the RTP-LLM OpenAI Compatible API server.

### Start with a model:

rtp-llm serve /mnt/nas1/hf/bge-reranker-base

### Specify the port:

rtp-llm serve /mnt/nas1/hf/bge-reranker-base --start_port 26000

### Check with --help for more options:

* **rtp-llm serve --help=listgroup**: To list all groups
* **rtp-llm serve --help=Concurrent**: To view a argument group
* **rtp-llm serve --help=concurrency_limit**: To view a single argument
* **rtp-llm serve --help=max**: To search by keyword
* **rtp-llm serve --help=page**: To view full help with pager (less/more)


See rtp-llm [serve](./../../backend/server_arguments.md) for the full reference of all available serve arguments.