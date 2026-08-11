#!/bin/bash
# /home/pi/bin/nuke — delete the currently playing song and skip to next

SOCKET=/tmp/mpvsocket

# Ask mpv for the full path of the current file
FILE=$(echo '{ "command": ["get_property", "path"] }' \
  | socat - $SOCKET | grep -o '"data":"[^"]*"' | cut -d'"' -f4)

if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
    echo "Couldn't determine current file"
    exit 1
fi

echo "Removing: $FILE"

# Skip first so mpv isn't sitting on the doomed track
echo 'playlist-next force' | socat - $SOCKET

# Remove it from mpv's playlist memory too, then delete from disk
rm "$FILE"