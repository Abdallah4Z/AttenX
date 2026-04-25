#!/usr/bin/env bash
# Script to download and prepare the CUB-200-2011 dataset for AttenX.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "Starting CUB dataset download..."

if ! command -v tar >/dev/null 2>&1; then
	echo "Error: tar is required but not installed." >&2
	exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
	echo "Error: unzip is required but not installed." >&2
	exit 1
fi

download_file() {
	local url="$1"
	local out="$2"

	if command -v curl >/dev/null 2>&1; then
		curl -fL "$url" -o "$out"
	else
		wget -O "$out" "$url"
	fi
}

download_google_drive_file() {
	local file_id="$1"
	local out="$2"

	if command -v gdown >/dev/null 2>&1; then
		gdown --id "$file_id" -O "$out"
		return
	fi

	# Fallback that handles Drive confirmation pages for larger files.
	local cookie_file
	local confirm_token
	local base_url
	cookie_file="$(mktemp)"
	base_url="https://drive.google.com/uc?export=download&id=${file_id}"

	confirm_token="$({ wget --quiet --save-cookies "$cookie_file" --keep-session-cookies "$base_url" -O - || true; } | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' | head -n1)"

	if [[ -n "$confirm_token" ]]; then
		wget --load-cookies "$cookie_file" "https://drive.google.com/uc?export=download&confirm=${confirm_token}&id=${file_id}" -O "$out"
	else
		wget "$base_url" -O "$out"
	fi

	rm -f "$cookie_file"
}

# Download CUB-200-2011 dataset images to data/CUB_200_2011
if [[ -d "CUB_200_2011" ]]; then
	echo "CUB_200_2011 already exists. Skipping image download."
else
	echo "Downloading images..."
	download_file "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1" "CUB_200_2011.tgz"
	tar -zxf "CUB_200_2011.tgz"
	rm -f "CUB_200_2011.tgz"
fi

# Download preprocessed metadata (train/test pickles)
if [[ -d "train" && -d "test" ]]; then
	echo "train/ and test/ already exist. Skipping metadata download."
else
	echo "Downloading captions/metadata..."
	metadata_zip="birds_text.zip"
	metadata_id="1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ"

	download_google_drive_file "$metadata_id" "$metadata_zip"

	if ! unzip -tq "$metadata_zip" >/dev/null 2>&1; then
		echo "Error: downloaded metadata archive is invalid or inaccessible." >&2
		echo "Try manually downloading from: https://drive.google.com/open?id=${metadata_id}" >&2
		rm -f "$metadata_zip"
		exit 1
	fi

	unzip -oq "$metadata_zip"
	rm -f "$metadata_zip"
fi

echo "Dataset downloaded successfully."
echo "Directory structure prepared for training in: $DIR"
