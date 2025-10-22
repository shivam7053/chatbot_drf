#!/usr/bin/env bash
set -e
python -m spacy download en_core_web_md
python -m spacy validate
