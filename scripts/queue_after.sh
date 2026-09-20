#!/bin/bash
# usage: queue_after.sh "<pattern>" <command...>
# waits until no process other than this helper (and its grep) matches the pattern, then runs the command
pat="$1"; shift
me=$$
while ps -eo pid,args | awk -v me="$me" '$1 != me' | grep -v -E "queue_after|grep" | grep -q -- "$pat"; do sleep 30; done
exec "$@"
