#!/bin/bash
# /home/pi/like — copy the currently playing song to the liked directory

SOCKET=/tmp/mpvsocket
LIKED=/home/pi/music-liked

FILE=$(echo '{ "command": ["get_property", "path"] }' \
  | socat - $SOCKET | grep -o '"data":"[^"]*"' | cut -d'"' -f4)

if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
    echo "Couldn't determine current file"
    exit 1
fi

mkdir -p "$LIKED"
cp -n "$FILE" "$LIKED/"     # -n = never overwrite if already liked
echo "Liked: $(basename "$FILE")"