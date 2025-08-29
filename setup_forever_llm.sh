#!/bin/bash

echo "Setting up ForeverLLM System..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download SpaCy model
echo "Downloading SpaCy language model..."
python -m spacy download en_core_web_sm

# Start Docker services
echo "Starting Docker services..."
docker-compose up -d

# Wait for Solr to be ready
echo "Waiting for Solr to start..."
sleep 30

# Upload LTR features to Solr
echo "Uploading LTR features to Solr..."
curl -X PUT 'http://localhost:8983/solr/forever_llm/schema/feature-store' \
  --data-binary @ltr_features.json \
  -H 'Content-type:application/json'

# Upload LTR model to Solr
echo "Uploading LTR model to Solr..."
curl -X PUT 'http://localhost:8983/solr/forever_llm/schema/model-store' \
  --data-binary @ltr_model.json \
  -H 'Content-type:application/json'

# Create data directory
mkdir -p forever_llm_data

echo "Setup complete! You can now run the ForeverLLM system."
echo "To start: python forever_llm.py"
EOF

chmod +x setup_forever_llm.sh
