@echo off
echo Running Pro TV Ingestion...
"C:\Users\David\Chatbot_Internet_Searches\venv\Scripts\python.exe" "C:\Users\David\Documents\UiPath\Stirile Pro TV\ingest.py"

echo Running Biziday Scraper...
"C:\Users\David\Chatbot_Internet_Searches\venv\Scripts\python.exe" "C:\Users\David\Chatbot_Internet_Searches\scrape_biziday.py"

echo Running Biziday Indexer...
"C:\Users\David\Chatbot_Internet_Searches\venv\Scripts\python.exe" "C:\Users\David\Chatbot_Internet_Searches\index_biziday.py"

echo All ingestions complete!
