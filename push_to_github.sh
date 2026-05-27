#!/bin/bash
# Run this once from your Terminal to push the week3 code to GitHub.
# The repo https://github.com/jaideep-aher/aipi561-week3 is already created.
#
# Usage:  bash push_to_github.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP_DIR="$(mktemp -d)"

echo "Copying files to $TMP_DIR ..."
cp -r "$SCRIPT_DIR/." "$TMP_DIR/"

# Remove any broken .git from the outputs folder copy
rm -rf "$TMP_DIR/.git"

cd "$TMP_DIR"

git init
git config user.email "jaideep.aher@duke.edu"
git config user.name "Jaideep Aher"
git branch -M main
git add -A
git commit -m "Week 3: data quality validation in CI/CD pipeline

- validation/check_data_quality.py: DataQualityValidator (4 issues: nulls, outliers, duplicates, dist-shift)
- validation/test_data_quality.py: full test suite — all synthetic tests pass
- .github/workflows/validate-data.yml: hourly CI + push-triggered validation
- backend/data.py: check_and_log_data_quality() graceful degradation at startup
- report.md: issues documented with impact + schedule justification"

git remote add origin https://github.com/jaideep-aher/aipi561-week3.git
echo ""
echo "Pushing to GitHub (you may be prompted for your GitHub username + token)..."
git push -u origin main

echo ""
echo "Done! View your repo at: https://github.com/jaideep-aher/aipi561-week3"
echo "GitHub Actions will run the validation workflow automatically."
