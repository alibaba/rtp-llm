#!/bin/bash

# The container must live as long as the supervisor, including its stop handler.
# A shell wrapper without exec can exit on TERM before its child finishes draining.
exec /bin/bash "/home/admin/${APP_NAME}/bin/startx.sh"
