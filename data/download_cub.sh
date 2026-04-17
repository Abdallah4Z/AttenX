#!/bin/bash
# Script to download and prepare the CUB-200-2011 dataset for AttenX (AttnGAN)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "Starting CUB dataset download..."

# Create birds directory
mkdir -p birds
cd birds

# Download CUB-200-2011 dataset images
echo "Downloading images..."
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -zxf CUB_200_2011.tgz
rm CUB_200_2011.tgz

# Download text/captions
echo "Downloading captions..."
# Note: For AttnGAN, the text embeddings and captions are usually provided as preprocessed pickles
# This URL is a common mirror for the text descriptions
wget https://drive.google.com/uc?export=download&id=1O_LtUP9sch09X3vQuFzX_vEUgM1P-Y3o -O birds_text.zip
unzip birds_text.zip
rm birds_text.zip

echo "Dataset downloaded successfully."
echo "Directory structure prepared for training."
