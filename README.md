# IR-project
a search engine for english  wikipidea  documents
Final project
Building a search engine for English Wikipedia
# Data
● Entire Wikipedia dump in a shared Google Storage bucket (same as Assignment #3).
● Pageviews for articles (you need to derive this; see code in Assignment #1 for details).
● Queries and a ranked list of up to 100 relevant results for them, split into train (30
queries+results given to you in queries_train.json) and test (held out for evaluation).
# Code
The staff provides you with the following pieces of code (available on Moodle):
● search_frontend.py: Flask app for search engine frontend. It has six blank methods there
that you need to implement.
● run_frontend_in_colab.ipynb: notebook showing how to run your search engine's frontend
in Colab for development purposes. The notebook also provides instructions for
querying/testing the engine.
● run_frontend_in_gcp.sh: command-line instructions for deploying your search engine to
GCP. You need to execute these commands to start a Compute Engine instance
(machine), reserve a public IP, and run your engine in GCP.
● startup_script_gcp.sh: a shell script that sets up the Compute Engine instance. No need to
modify this unless you need additional packages installed in GCP.

