#!/bin/bash
echo 'cleaning'

while read p; do
  echo 'Deleting ' $p
  rm -rf .aws-sam/build/BeAwesomeDevChatBotFunction/$p
done < .samignore