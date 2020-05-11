#!/usr/bin/env bash

echo "Installing hooks..."
cp pre-push.sh .git/hooks/pre-push
echo "Done!"
