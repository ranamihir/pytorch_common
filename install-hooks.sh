#!/usr/bin/env bash

GIT_DIR=$(git rev-parse --git-dir)

echo "Installing hooks..."
# this command creates symlink to our pre-push script
ln -fs ../../pre-push.sh $GIT_DIR/hooks/pre-push
echo "Done!"
