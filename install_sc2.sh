#!/bin/bash

# Install StarCraft II for SMAC
# This script downloads and installs StarCraft II headless version

SC2_VERSION="4.10"
SC2_PATH="$HOME/StarCraftII"

echo "Installing StarCraft II version $SC2_VERSION..."

# Create installation directory
mkdir -p $SC2_PATH

# Download StarCraft II
cd $SC2_PATH
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    echo "Downloading StarCraft II for macOS..."
    wget -nc http://blzdistsc2-a.akamaihd.net/Mac/SC2.$SC2_VERSION.zip
    unzip -P iagreetotheeula SC2.$SC2_VERSION.zip
    rm SC2.$SC2_VERSION.zip
else
    # Linux
    echo "Downloading StarCraft II for Linux..."
    wget -nc http://blzdistsc2-a.akamaihd.net/Linux/SC2.$SC2_VERSION.zip
    unzip -P iagreetotheeula SC2.$SC2_VERSION.zip
    rm SC2.$SC2_VERSION.zip
fi

# Download SMAC maps
echo "Downloading SMAC maps..."
MAP_DIR="$SC2_PATH/Maps"
mkdir -p $MAP_DIR
cd $MAP_DIR

wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
rm SMAC_Maps.zip

echo "StarCraft II installation complete!"
echo "SC2 installed at: $SC2_PATH"
