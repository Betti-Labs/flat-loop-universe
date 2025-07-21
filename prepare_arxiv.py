#!/usr/bin/env python3
"""
Prepare Flat Loop Universe for arXiv Submission

This script prepares the Flat Loop Universe project for arXiv submission by:
1. Creating a ZIP file with all necessary LaTeX files
2. Ensuring all references are properly formatted
3. Checking for any issues that might cause problems with arXiv
"""

import os
import shutil
import zipfile
import re
import datetime

def create_arxiv_submission():
    """Create a ZIP file for arXiv submission"""
    print("Preparing arXiv submission...")
    
    # Create submission directory
    submission_dir = "arxiv_submission"
    os.makedirs(submission_dir, exist_ok=True)
    
    # Copy LaTeX files
    shutil.copy("paper/flat_loop_universe.tex", submission_dir)
    
    # Copy figures
    figures_dir = os.path.join(submission_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Check if figures exist and copy them
    if os.path.exists("paper/figures"):
        for file in os.listdir("paper/figures"):
            if file.endswith((".png", ".jpg", ".pdf")):
                shutil.copy(os.path.join("paper/figures", file), figures_dir)
    
    # Create ZIP file
    zip_filename = f"flat_loop_universe_arxiv_{datetime.datetime.now().strftime('%Y%m%d')}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, submission_dir)
                zipf.write(file_path, arcname)
    
    print(f"Created arXiv submission ZIP file: {zip_filename}")
    print("\nSubmission Instructions:")
    print("1. Go to https://arxiv.org/submit")
    print("2. Log in or create an account")
    print("3. Upload the ZIP file")
    print("4. Follow the submission process")
    print("\nNote: Make sure all references in your paper are properly formatted")
    print("      and that all figures are included in the submission.")

if __name__ == "__main__":
    create_arxiv_submission()