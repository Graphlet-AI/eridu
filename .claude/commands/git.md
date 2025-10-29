---
allowed-tools: Bash(git checkout --branch:*), Bash(git status:*), Bash(git commit:*), Bash(gh pr create:*), Bash(pre-commit), Edit, Read, Write
description: Prepare and commit the code already added to git
---

Run pre-commit to find any outstanding problems with the changes already added to git. Once they are taken care of, you may git add only the original files I already added to re-run pre-commit. DO NOT add any other files to git. When pre-commit passes, commit the code. Don't use 'pre-commit run --all-files' to validate the code, just run 'pre-commit' to run it on files that are already git added. Add them before running pre-commit if you have made changes. If black says it reformatted a file, add it and then re-run pre-commit.
