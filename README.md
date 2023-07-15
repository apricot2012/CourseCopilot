# Course Copilot

### Setting up OpenAI APIs
Put your openAI API key in ``constants.py SECRET_KEY``

### Setting up OpenSearch
Create the following yml file:
https://opensearch.org/docs/latest/install-and-configure/install-opensearch/docker/#sample-docker-compose-file-for-development

run ``sysctl -w vm.max_map_count=262144``
run ``docker-compose up``

### Running the system
Run ``main.py``