#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-data/abo_raw}
mkdir -p "$ROOT"

echo "Downloading ABO metadata and spin archive into $ROOT"

# Product listings / metadata
aws s3 cp --no-sign-request \
  s3://amazon-berkeley-objects/abo-listings.tar \
  "$ROOT/abo-listings.tar"

# 360-degree spin images and metadata.
# This is about 40 GB according to the official download page.
aws s3 cp --no-sign-request \
  s3://amazon-berkeley-objects/abo-spins.tar \
  "$ROOT/abo-spins.tar"

echo "Extracting..."
tar -xf "$ROOT/abo-listings.tar" -C "$ROOT"
tar -xf "$ROOT/abo-spins.tar" -C "$ROOT"

echo "Done."