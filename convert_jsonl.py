#!/usr/bin/env python3
"""
JSONL to JSON Converter
Converts JSON Lines (.jsonl) files to standard JSON format
"""

import json
import os
import sys
from pathlib import Path

def convert_jsonl_to_json(input_file, output_file=None, output_dir=None):
    """
    Convert a JSONL file to standard JSON format
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSON file (optional)
        output_dir: Directory where to save output files (optional)
    """
    input_path = Path(input_file)
    
    # Check if file exists
    if not input_path.exists():
        print(f" Error: File '{input_file}' not found")
        return False
    
    # Determine output file path
    if output_file is None:
        # Remove .jsonl extension if present, or just use stem
        if input_path.suffix == '.jsonl':
            output_filename = input_path.stem + '.json'
        else:
            output_filename = input_path.stem + '_converted.json'
        
        # Use output_dir if specified, otherwise same directory as input
        if output_dir:
            output_path = Path(output_dir) / output_filename
        else:
            output_path = input_path.parent / output_filename
    else:
        # If output_file is specified, use it as is (don't apply output_dir)
        output_path = Path(output_file)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Read JSONL file
        entries = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Error parsing line {line_num}: {e}")
                        continue
        
        # Write standard JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        
        print(f"Converted: {input_file}")
        print(f"   → Output: {output_path}")
        print(f"   → Entries: {len(entries)}")
        return True
        
    except Exception as e:
        print(f" Error converting {input_file}: {e}")
        return False


def convert_directory(directory, pattern='*.jsonl', recursive=False, output_dir=None):
    """
    Convert all JSONL files in a directory
    
    Args:
        directory: Directory to search
        pattern: File pattern to match (default: *.jsonl)
        recursive: Search recursively in subdirectories
        output_dir: Directory where to save converted files (optional)
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f" Error: Directory '{directory}' not found")
        return
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f" Output directory: {output_path.absolute()}\n")
    
    # Find files
    if recursive:
        files = list(dir_path.rglob(pattern))
    else:
        files = list(dir_path.glob(pattern))
    
    if not files:
        print(f" No files matching '{pattern}' found in {directory}")
        return
    
    print(f"Found {len(files)} file(s) to convert\n")
    
    success_count = 0
    for file_path in files:
        # Skip if it's already a converted file
        if file_path.stem.endswith('_converted'):
            print(f"  Skipping: {file_path.name} (already converted)")
            continue
        
        # Determine output name: remove .jsonl and add .json
        if file_path.suffix == '.jsonl':
            output_filename = file_path.stem + '.json'
        else:
            output_filename = file_path.stem + '_converted.json'
        
        # Determine output location
        if output_dir:
            output_file = Path(output_dir) / output_filename
        else:
            output_file = file_path.parent / output_filename
        
        # Skip if output file already exists
        if output_file.exists():
            print(f"  Skipping: {file_path.name} (output file already exists: {output_file.name})")
            continue
            
        if convert_jsonl_to_json(file_path, output_file=str(output_file)):
            success_count += 1
        print()
    
    print(f"\n{'='*60}")
    print(f"Conversion complete: {success_count}/{len(files)} files converted")
    if output_dir:
        print(f"All files saved to: {Path(output_dir).absolute()}")


def main():
    """Main function to handle command line arguments"""
    
    if len(sys.argv) < 2:
        print("JSONL to JSON Converter")
        print("="*60)
        print("\nUsage:")
        print("  Single file:")
        print("    python convert_jsonl.py <input_file> [output_file]")
        print("    python convert_jsonl.py <input_file> --output-dir <directory>")
        print("\n  Directory:")
        print("    python convert_jsonl.py --dir <directory> [options]")
        print("\n  Options:")
        print("    --pattern <pattern>        File pattern (default: *.jsonl)")
        print("    --recursive                Search recursively in subdirectories")
        print("    --output-dir <directory>   Save converted files to this directory")
        print("\nExamples:")
        print("  python convert_jsonl.py queries.jsonl")
        print("  python convert_jsonl.py queries.jsonl output.json")
        print("  python convert_jsonl.py queries.jsonl --output-dir ./converted")
        print("  python convert_jsonl.py --dir ./data")
        print("  python convert_jsonl.py --dir ./data --output-dir ./converted")
        print("  python convert_jsonl.py --dir ./data --pattern '*.jsonl' --recursive --output-dir ./json_files")
        print("\nNote:")
        print("  - If --output-dir is NOT specified, files are saved in the same directory as the input")
        print("  - If --output-dir IS specified, files are saved ONLY in the output directory")
        sys.exit(1)
    
    # Directory mode
    if sys.argv[1] == '--dir':
        if len(sys.argv) < 3:
            print(" Error: Please specify a directory")
            sys.exit(1)
        
        directory = sys.argv[2]
        pattern = '*.jsonl'  # Default to .jsonl files
        recursive = False
        output_dir = None
        
        # Parse optional arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == '--pattern' and i + 1 < len(sys.argv):
                pattern = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--recursive':
                recursive = True
                i += 1
            elif sys.argv[i] == '--output-dir' and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        convert_directory(directory, pattern, recursive, output_dir)
    
    # Single file mode
    else:
        input_file = sys.argv[1]
        output_file = None
        output_dir = None
        
        # Parse optional arguments
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == '--output-dir' and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
                i += 2
            else:
                # If no flag, assume it's the output file
                if output_file is None and not sys.argv[i].startswith('--'):
                    output_file = sys.argv[i]
                i += 1
        
        convert_jsonl_to_json(input_file, output_file, output_dir)


if __name__ == '__main__':
    main()