#!/bin/bash

USER_ID=${LOCAL_UID:-62000}
GROUP_ID=${LOCAL_GID:-62000}

echo "Starting with UID : $USER_ID, GID: $GROUP_ID"
useradd -u $USER_ID -o -m user
groupmod -g $GROUP_ID user
export HOME=/home/user

/linux-nvidia/run > /dev/null 2>&1 &

touch $HOME/.zundavox.yaml

exec /usr/sbin/gosu user /bin/bash