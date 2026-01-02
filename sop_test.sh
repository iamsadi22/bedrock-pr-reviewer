#!/bin/bash
# Helper script to run Python SOP retriever for a diff chunk

DIFF_CHUNK="$1"
python3 src/sop_chunk_test.py "$DIFF_CHUNK"
