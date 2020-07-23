#!/bin/sh

# The tests need to be run from within an environment that has pytorch installed (of course).
# A `pytorch_common` conda env with the required dependencies may be installed by cloning this
# repo and running:
# `conda env create -f requirements-dev.yaml`
# For running all the tests, you also need to have the `transformers` repo installed, otherwise
# those will be skipped. This may be installed by running from within the `pytorch_common` env:
# `pip install .[nlp]"`

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

current_branch=`git branch | grep '*' | sed 's/* //'`

### Uncomment the following code block to run tests only on master branch ###
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


### Comment the following code block to prevent running tests on all branches ###
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
