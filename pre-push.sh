#!/bin/sh

# Check for python breakpoints
alias pythonfiles='find . -type f -name "*.py" | xargs git diff --cached --name-only $against'
if [ -n "$(pythonfiles)" ]; then
  if pythonfiles | xargs grep --color --with-filename -n "import pdb"; then
    echo "Error pushing changes: Please remove pdb and its usage."
    exit 1
  fi
  if pythonfiles | xargs grep --color --with-filename -n "pdb.set_trace()"; then
    echo "Error pushing changes: Please remove pdb and its usage."
    exit 1
  fi
  if pythonfiles | xargs grep --color --with-filename -n "breakpoint()"; then
    echo "Error pushing changes: Please remove breakpoint and its usage."
    exit 1
  fi
fi

# Activate env where pytorch is installed
source activate pytorch_common

current_branch=`git branch | grep '*' | sed 's/* //'`

# if [ "$current_branch" = "master" ]; then
    # echo "You are about to push to master. Running all tests first..."
#     python -m unittest discover tests
#     if [ $? -eq 0 ]; then
#         # Tests passed; proceed to prepare push message
#         echo "All tests passed!"
#         exit 0
#     else
#         # Some tests failed; prevent from pushing broken code to master
#         echo "Some tests failed! Pushing broken code to master is not allowed! Aborting the push."
#         echo "Note: You can still push broken code to feature branches."
#         exit 1
#     fi
# fi


echo "Running all tests..."
python -m unittest discover tests
if [ $? -eq 0 ]; then
    # Tests passed; proceed to prepare push message
    echo "All tests passed!"
    exit 0
else
    # Some tests failed; prevent from pushing broken code
    echo "Some tests failed! Pushing broken code is not allowed! Aborting the push."
    exit 1
fi
