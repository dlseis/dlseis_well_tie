#!/bin/bash
echo "Build wtie package documentation"
pdoc --html --output-dir documentation --config show_source_code=False --force wtie
