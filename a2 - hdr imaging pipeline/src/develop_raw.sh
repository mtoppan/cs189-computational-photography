#!/bin/bash
# find ../data/door_stack -name "*.nef" -exec sh -c 'dcraw -v -T -4 -w -o 1 "$1" 2>&1' sh {} \; > output.txt
find ../data/dali_stack -name "*.cr2" -exec sh -c 'dcraw -v -T -4 -w -o 1 "$1" 2>&1' sh {} \; > outputp4.txt