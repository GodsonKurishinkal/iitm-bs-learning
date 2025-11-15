# Git Commands Cheatsheet

## Setup

```bash
# Configure user
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# View config
git config --list
```

## Creating Repositories

```bash
# Initialize new repo
git init

# Clone existing repo
git clone <url>
git clone <url> <directory-name>
```

## Basic Workflow

```bash
# Check status
git status

# Add files to staging
git add <file>              # specific file
git add .                   # all files in current directory
git add -A                  # all files in repo

# Commit changes
git commit -m "Commit message"
git commit -am "Message"    # add and commit tracked files

# View history
git log
git log --oneline           # compact view
git log --graph             # visual graph
```

## Branching

```bash
# List branches
git branch
git branch -a               # including remote

# Create branch
git branch <branch-name>

# Switch branch
git checkout <branch-name>
git switch <branch-name>    # newer command

# Create and switch
git checkout -b <branch-name>
git switch -c <branch-name>

# Delete branch
git branch -d <branch-name>
git branch -D <branch-name> # force delete

# Rename branch
git branch -m <old-name> <new-name>
```

## Merging

```bash
# Merge branch into current
git merge <branch-name>

# Abort merge
git merge --abort
```

## Remote Repositories

```bash
# List remotes
git remote -v

# Add remote
git remote add origin <url>

# Push to remote
git push origin <branch>
git push -u origin <branch> # set upstream

# Pull from remote
git pull origin <branch>
git pull                    # if upstream set

# Fetch changes
git fetch origin
```

## Undoing Changes

```bash
# Unstage file
git reset <file>
git restore --staged <file>

# Discard changes in working directory
git checkout -- <file>
git restore <file>

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Revert commit (create new commit)
git revert <commit-hash>
```

## Viewing Changes

```bash
# Show unstaged changes
git diff

# Show staged changes
git diff --staged
git diff --cached

# Show changes between commits
git diff <commit1> <commit2>

# Show changes in specific file
git diff <file>
```

## Stashing

```bash
# Stash changes
git stash
git stash save "message"

# List stashes
git stash list

# Apply stash
git stash apply
git stash apply stash@{n}

# Apply and remove stash
git stash pop

# Drop stash
git stash drop stash@{n}

# Clear all stashes
git stash clear
```

## Tagging

```bash
# Create tag
git tag <tag-name>
git tag -a <tag-name> -m "message"

# List tags
git tag

# Push tags
git push origin <tag-name>
git push origin --tags      # all tags

# Delete tag
git tag -d <tag-name>
git push origin :refs/tags/<tag-name>  # remote
```

## Common Workflows

### Feature Branch Workflow
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add new feature"

# Push to remote
git push -u origin feature/new-feature

# Merge into main
git checkout main
git merge feature/new-feature
git push origin main

# Delete feature branch
git branch -d feature/new-feature
git push origin --delete feature/new-feature
```

### Updating Local Repository
```bash
# Fetch and merge
git fetch origin
git merge origin/main

# Or use pull (fetch + merge)
git pull origin main
```

### Fixing Last Commit
```bash
# Amend last commit message
git commit --amend -m "New message"

# Add file to last commit
git add <forgotten-file>
git commit --amend --no-edit
```

## Useful Aliases

Add to `~/.gitconfig`:

```
[alias]
    st = status
    co = checkout
    br = branch
    ci = commit
    unstage = reset HEAD --
    last = log -1 HEAD
    lg = log --graph --oneline --all
```

## Tips

- Always `git pull` before starting new work
- Commit often with meaningful messages
- Use branches for features/experiments
- Review changes before committing (`git diff`)
- Don't commit sensitive data (use `.gitignore`)
